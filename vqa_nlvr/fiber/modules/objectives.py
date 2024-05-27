import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from .dist_utils import all_gather
import torch.distributed as dist


def compute_vqa(pl_module, batch):
    if "inter_modality" in pl_module.exp_name:
        infer = pl_module.infer(batch)
        vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    elif "image_unimodal" in pl_module.exp_name:
       infer = pl_module.infer(batch, image_only=True)
       vqa_logits = pl_module.vqa_classifier(infer["cls_feats"]) 
    elif "text_unimodal" in pl_module.exp_name:
       infer = pl_module.infer(batch, text_only=True)
       vqa_logits = pl_module.vqa_classifier(infer["cls_feats"]) 
    elif "intra_modality" in pl_module.exp_name:
        infer_text = pl_module.infer(batch, text_only=True)
        infer_image = pl_module.infer(batch, image_only=True)
        vqa_logits = pl_module.vqa_classifier(infer_image["cls_feats"],
                                              infer_text["cls_feats"]
                                             )
    else:
        infer = pl_module.infer(batch)
        infer_text = pl_module.infer(batch, text_only=True)
        infer_image = pl_module.infer(batch, image_only=True)
        vqa_logits, vqa_logits_cat, vqa_logits_image, vqa_logits_text = pl_module.vqa_classifier(infer["cls_feats"],
                                                                                                 infer_image["cls_feats"],
                                                                                                 infer_text["cls_feats"]
                                                                                                 )
    vqa_targets = torch.zeros(len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets.float()) * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    if "inter_and_intra_multimodal" in pl_module.exp_name:
        if "ensemble" in  pl_module.exp_name:
            ret = {
            "vqa_loss": vqa_loss,
            "vqa_logits": vqa_logits,
            "vqa_targets": vqa_targets,
            "vqa_labels": vqa_labels,
            "vqa_scores": vqa_scores,
            "image": infer["image"],
            "text": infer["text"],
       }
        else:
            ret = {
            "vqa_loss": vqa_loss,
            "vqa_logits": vqa_logits,
            "vqa_logits_cat": vqa_logits_cat,
            "vqa_logits_image": vqa_logits_image,
            "vqa_logits_text": vqa_logits_text,
            "vqa_targets": vqa_targets,
            "vqa_labels": vqa_labels,
            "vqa_scores": vqa_scores,
            "image": infer["image"],
            "text": infer["text"],
       }
    else:
        ret = {
            "vqa_loss": vqa_loss,
            "vqa_logits": vqa_logits,
            "vqa_targets": vqa_targets,
            "vqa_labels": vqa_labels,
            "vqa_scores": vqa_scores,
            "image": infer["image"],
            "text": infer["text"],
        }


    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(ret["vqa_logits"], ret["vqa_targets"])
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_nlvr2(pl_module, batch):
    if "inter_modality" in pl_module.exp_name:
        embeds1 = pl_module.infer(batch, image_token_type_idx=1)
        embeds2 = pl_module.infer(batch, image_token_type_idx=2)
        cls_feats = torch.cat([embeds1["cls_feats"], embeds2["cls_feats"]], dim=-1)
        nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)
    elif "image_unimodal" in pl_module.exp_name:
        embeds1 = pl_module.infer(batch, image_only=True, image_token_type_idx=1)
        embeds2 = pl_module.infer(batch, image_only=True, image_token_type_idx=2)
        x_image = torch.cat([embeds1["cls_feats"], embeds2["cls_feats"]], dim=-1)
        nlvr2_logits = pl_module.nlvr2_classifier(x_image) 
    elif "text_unimodal" in pl_module.exp_name:
       x_text = pl_module.infer(batch, text_only=True)["cls_feats"]
       nlvr2_logits = pl_module.nlvr2_classifier(x_text) 
    elif "intra_modality" in pl_module.exp_name:
        embeds1 = pl_module.infer(batch, image_only=True, image_token_type_idx=1)
        embeds2 = pl_module.infer(batch, image_only=True, image_token_type_idx=2)
        x_image = torch.cat([embeds1["cls_feats"], embeds2["cls_feats"]], dim=-1)
        x_text = pl_module.infer(batch, text_only=True)["cls_feats"]
        nlvr2_logits = pl_module.nlvr2_classifier(x_image,
                                                  x_text
                                                 )
    else:
        embeds1 = pl_module.infer(batch, image_only=True, image_token_type_idx=1)
        embeds2 = pl_module.infer(batch, image_only=True, image_token_type_idx=2)
        x_image = torch.cat([embeds1["cls_feats"], embeds2["cls_feats"]], dim=-1)
        x_text = pl_module.infer(batch, text_only=True)["cls_feats"]

        cat_embeds1 = pl_module.infer(batch, image_token_type_idx=1)
        cat_embeds2 = pl_module.infer(batch, image_token_type_idx=2)
        cls_feats = torch.cat([cat_embeds1["cls_feats"], cat_embeds2["cls_feats"]], dim=-1)

        nlvr2_logits, nlvr2_logits_cat, nlvr2_logits_image, nlvr2_logits_text = pl_module.nlvr2_classifier(cls_feats,
                                                                                                 x_image,
                                                                                                 x_text,
                                                                                                 )

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels.view(-1))
# 

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_logits_image": nlvr2_logits_image,
        "nlvr2_logits_text": nlvr2_logits_text,
        "nlvr2_logits_cat": nlvr2_logits_cat,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(ret["nlvr2_logits"], ret["nlvr2_labels"])
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "val" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"val_nlvr2_loss")(
                F.cross_entropy(ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches])
            )
            dev_acc = getattr(pl_module, f"val_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/val/loss", dev_loss)
            pl_module.log(f"nlvr2/val/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"{phase}_nlvr2_loss")(
                F.cross_entropy(ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches])
            )
            test_acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    if "intra_modality" in pl_module.exp_name:
        return {"qids": qids, "preds": vqa_preds,
                "vqa_logits": output["vqa_logits"],
                "vqa_targets": output["vqa_targets"],
                "image": output["image"],
                "text": output["text"],
                }
    elif "inter_and_intra_multimodal" in pl_module.exp_name: 
        return {"qids": qids, "preds": vqa_preds,
                "vqa_logits": output["vqa_logits"],
                "logits_cat": output["vqa_logits_cat"], 
                "logits_image": output["vqa_logits_image"], 
                "logits_text": output["vqa_logits_text"],
                "vqa_targets": output["vqa_targets"],
                "image": output["image"],
                "text": output["text"],
                }
    else:
        return {"qids": qids, "preds": vqa_preds,
                "vqa_logits": output["vqa_logits"],
                "vqa_targets": output["vqa_targets"],
                "image": output["image"],
                "text": output["text"],
                }
        

def arc_test_step(pl_module, batch, output):
    return output


def get_vqa_score(logits, target):
    logits, target = (
            logits.detach().float(),
            target.detach().float(),
    )
    logits = torch.max(logits, 1)[1]
    one_hots = torch.zeros(*target.size()).to(target)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * target
    return scores.sum() / len(logits)

def get_vqa_score_logits(logits, target):
    logits, target = (
            logits.detach().float(),
            target.detach().float(),
    )
    logits = torch.max(logits, 1)[1]
    one_hots = torch.zeros(*target.size()).to(target)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * target
    return torch.sum(scores, dim=1)


def vqa_test_wrapup(pl_module, outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]
    
    if "inter_and_intra_modality" in pl_module.exp_name and ("intra_modality" not in pl_module.exp_name):
        targets, logits = list(), list()
        logits_image, logits_text, logits_cat = list(), list(), list()
        for out in outs:
            targets += out["vqa_targets"]
            logits += out["vqa_logits"]
            logits_image += out["logits_image"]
            logits_text += out["logits_text"]
            logits_cat += out["logits_cat"]


        targets = torch.cat(targets, dim=0)
        logits_image = torch.cat(logits_image, dim=0)
        logits_text = torch.cat(logits_text, dim=0)
        logits_cat = torch.cat(logits_cat, dim=0)
        logits = torch.cat(logits, dim=0)
        if len(logits_image.shape) > 2 : # if using multiple gpus, a new dimension (of size num_gpus) is added after all_gather
            targets = targets.reshape(targets.shape[0] * targets.shape[1], targets.shape[2])
            logits_image = logits_image.reshape(logits_image.shape[0] * \
                                                 logits_image.shape[1], logits_image.shape[2])
            logits_text = logits_text.reshape(logits_text.shape[0] * \
                                                 logits_text.shape[1], logits_text.shape[2]) 
            logits_cat = logits_cat.reshape(logits_cat.shape[0] * \
                                                 logits_cat.shape[1], logits_cat.shape[2]) 
            logits = logits.reshape(logits.shape[0] * \
                                                 logits.shape[1], logits.shape[2]) 

        optimal_score = get_vqa_score(logits, targets)

        print(f"VQA score: {optimal_score}")
