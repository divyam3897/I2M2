import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import copy
import matplotlib.pyplot as plt

from . import swin_transformer, roberta
from . import heads, objectives, fiber_utils
from .swin_helpers import swin_adapt_position_encoding
from transformers import RobertaConfig
from .roberta import RobertaModel
from .classify import InterModalityClassifier, UnimodalClassifier, InterAndIntraModalityClassifier, IntraModalityClassifier

@torch.no_grad()
def concat_all_gather(tensor):
    # from albef
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class FIBERTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.exp_name = self.hparams.config["exp_name"]

        bert_config = RobertaConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        resolution_after = config["image_size"]
        self.num_fuse_block = config["num_fuse_block"]
        self.num_text_layer = config["num_layers"]
        roberta.NUM_FUSE_BLOCK = swin_transformer.NUM_FUSE_BLOCK = self.num_fuse_block
        roberta.DIM_IMG = config["input_image_embed_size"]
        swin_transformer.DIM_TXT = config["input_text_embed_size"]

        self.cross_modal_text_transform = nn.Linear(config["input_text_embed_size"], config["hidden_size"])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(config["input_image_embed_size"], config["hidden_size"])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        self.cross_modal_text_transform_itc = nn.Linear(config["input_text_embed_size"], config["hidden_size"])
        self.cross_modal_text_transform_itc.apply(objectives.init_weights)
        self.cross_modal_image_transform_itc = nn.Linear(config["input_image_embed_size"], config["hidden_size"])
        self.cross_modal_image_transform_itc.apply(objectives.init_weights)

        # create the queue from ALBEF
        if config["loss_names"]["itc"] > 0:
            self.temp = nn.Parameter(torch.ones([]) * 0.07)
            self.queue_size = 4096
            self.register_buffer("image_queue", torch.randn(config["hidden_size"], self.queue_size))
            self.register_buffer("text_queue", torch.randn(config["hidden_size"], self.queue_size))
            self.register_buffer("image_input_queue", torch.randn(self.queue_size, 3, config['image_size'], config['image_size']))
            self.register_buffer("text_input_queue", torch.zeros(self.queue_size, config["max_text_len"], dtype=torch.long))
            self.register_buffer("text_input_mask_queue", torch.zeros(self.queue_size, config["max_text_len"], dtype=torch.long))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
            self.register_buffer("queue_total", torch.zeros(1, dtype=torch.long))  

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                getattr(swin_transformer, self.hparams.config["vit"])(
                    pretrained=config["pretrained_vit"],
                    config=self.hparams.config,
                )
                RobertaModel.from_pretrained(config["tokenizer"])

            torch.distributed.barrier()

        self.vit_model = getattr(swin_transformer, self.hparams.config["vit"])(
            pretrained=config["pretrained_vit"],
            config=self.hparams.config,
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.text_transformer = RobertaModel.from_pretrained(config["tokenizer"])

        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)
        self.itc_pooler = config["itc_pooler"]
        if self.itc_pooler:
            self.cross_modal_image_pooler_itc = heads.Pooler(config["hidden_size"])
            self.cross_modal_image_pooler_itc.apply(objectives.init_weights)
            self.cross_modal_text_pooler_itc = heads.Pooler(config["hidden_size"])
            self.cross_modal_text_pooler_itc.apply(objectives.init_weights)

        if (
            config["loss_names"]["mlm"] > 0
            or config["loss_names"]["caption_mle"] > 0
            or config["loss_names"]["caption_gold"] > 0
            or config["loss_names"]["caption_cider"] > 0
        ):
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)


        hs = self.hparams.config["hidden_size"]

        
        # ===================== Downstream ===================== #
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            for key in ['image_queue', 'text_queue', 'queue_ptr', 'queue_total', 'image_input_queue', 'text_input_queue', 'text_input_mask_queue']:
                if key in state_dict:
                    state_dict.pop(key)
            state_dict = swin_adapt_position_encoding(
                state_dict, before=config["resolution_before"], after=resolution_after
            )
            self.load_state_dict(state_dict, strict=False)

        if self.hparams.config["loss_names"]["vqa"] > 0:
            y_dim = self.hparams.config["vqav2_label_size"]
            if "inter_modality" in self.hparams.config["exp_name"]:
                self.vqa_classifier = InterModalityClassifier(self.hparams.config["hidden_size"], y_dim, self.hparams.config["exp_name"])
            elif "unimodal" in self.hparams.config["exp_name"]:
                self.vqa_classifier = UnimodalClassifier(self.hparams.config["hidden_size"], y_dim, self.hparams.config["exp_name"]) 
            elif "intra_modality" in self.hparams.config["exp_name"]:
                self.vqa_classifier = IntraModalityClassifier(self.hparams.config["hidden_size"], y_dim, self.hparams.config["exp_name"]) 
            else:
                self.vqa_classifier = InterAndIntraModalityClassifier(self.hparams.config["hidden_size"], y_dim, self.hparams.config["exp_name"])
            

            if ("inter_and_intra_modality_vqa" in self.hparams.config["exp_name"]) and (not config['test_only']):
                weight_layer_names = ['vqa_classifier.model.0.weight', 'vqa_classifier.model.1.weight', '', 'vqa_classifier.model.3.weight']
                bias_layer_names = ['vqa_classifier.model.0.bias', 'vqa_classifier.model.1.bias', '', 'vqa_classifier.model.3.bias']
            
                ckpt = torch.load('./ckpts/image/1/best_epoch=15-val/the_metric=0.26.ckpt', map_location="cpu")
                image_state_dict = ckpt["state_dict"]
                keys_to_remove = [key for key in image_state_dict.keys() if not key.startswith('vqa')]
                for key in keys_to_remove:
                    image_state_dict.pop(key)

                for idx, layer in enumerate(self.vqa_classifier.image_model):
                    for name, param in layer.named_parameters():
                        if name == "weight" and idx != 2:
                            param.data = image_state_dict[weight_layer_names[idx]].clone()
                        elif name == "bias" and idx != 2:
                            param.data = image_state_dict[bias_layer_names[idx]].clone()

                del image_state_dict, ckpt

                ckpt = torch.load('./ckpts/text/1/best_epoch=35-val/the_metric=0.44.ckpt', map_location="cpu")
                text_state_dict = ckpt["state_dict"]
                keys_to_remove = [key for key in text_state_dict.keys() if not key.startswith('vqa')]
                for key in keys_to_remove:
                    text_state_dict.pop(key)

                for idx, layer in enumerate(self.vqa_classifier.text_model):
                    for name, param in layer.named_parameters():
                        if name == "weight" and idx != 2:
                            param.data = text_state_dict[weight_layer_names[idx]].clone()
                        elif name == "bias" and idx != 2:
                            param.data = text_state_dict[bias_layer_names[idx]].clone()

                del text_state_dict, ckpt

                ckpt = torch.load('./ckpts/inter_modality/1/best_epoch=97-val/the_metric=0.68.ckpt', map_location="cpu")
                md_state_dict = ckpt["state_dict"]
                keys_to_remove = [key for key in md_state_dict.keys() if not key.startswith('vqa')]
                for key in keys_to_remove:
                    md_state_dict.pop(key)

                for idx, layer in enumerate(self.vqa_classifier.cat_model):
                    for name, param in layer.named_parameters():
                        if name == "weight" and idx != 2:
                            param.data = md_state_dict[weight_layer_names[idx]].clone()
                        elif name == "bias" and idx != 2:
                            param.data = md_state_dict[bias_layer_names[idx]].clone()

                del md_state_dict, ckpt
            
            if ("intra_modality" in self.hparams.config["exp_name"]) and (not config['test_only']):
                weight_layer_names = ['vqa_classifier.model.0.weight', 'vqa_classifier.model.1.weight', '', 'vqa_classifier.model.3.weight']
                bias_layer_names = ['vqa_classifier.model.0.bias', 'vqa_classifier.model.1.bias', '', 'vqa_classifier.model.3.bias']
            
                ckpt = torch.load('./ckpts/image/1/best_epoch=08-val/the_metric=0.26.ckpt', map_location="cpu")
                image_state_dict = ckpt["state_dict"]
                keys_to_remove = [key for key in image_state_dict.keys() if not key.startswith('vqa')]
                for key in keys_to_remove:
                    image_state_dict.pop(key)

                for idx, layer in enumerate(self.vqa_classifier.image_model):
                    for name, param in layer.named_parameters():
                        if name == "weight" and idx != 2:
                            param.data = image_state_dict[weight_layer_names[idx]].clone()
                        elif name == "bias" and idx != 2:
                            param.data = image_state_dict[bias_layer_names[idx]].clone()

                del image_state_dict, ckpt

                ckpt = torch.load('./ckpts/text/1/best_epoch=54-val/the_metric=0.44.ckpt', map_location="cpu")
                text_state_dict = ckpt["state_dict"]
                keys_to_remove = [key for key in text_state_dict.keys() if not key.startswith('vqa')]
                for key in keys_to_remove:
                    text_state_dict.pop(key)

                for idx, layer in enumerate(self.vqa_classifier.text_model):
                    for name, param in layer.named_parameters():
                        if name == "weight" and idx != 2:
                            param.data = text_state_dict[weight_layer_names[idx]].clone()
                        elif name == "bias" and idx != 2:
                            param.data = text_state_dict[bias_layer_names[idx]].clone()

                del text_state_dict, ckpt
            else:
                self.vqa_classifier.apply(objectives.init_weights)

        
        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            y_dim = self.hparams.config["vqav2_label_size"]
            if "inter_modality" in self.hparams.config["exp_name"]:
                self.nlvr2_classifier = InterModalityClassifier(self.hparams.config["hidden_size"], output_dim=2, exp_name=self.hparams.config["exp_name"])
            elif "unimodal" in self.hparams.config["exp_name"]:
                self.nlvr2_classifier = UnimodalClassifier(self.hparams.config["hidden_size"], output_dim=2, exp_name=self.hparams.config["exp_name"]) 
            elif "intra_modality" in self.hparams.config["exp_name"]:
                self.nlvr2_classifier = IntraModalityClassifier(self.hparams.config["hidden_size"], output_dim=2, exp_name=self.hparams.config["exp_name"]) 
            else:
                self.nlvr2_classifier = InterAndIntraModalityClassifier(self.hparams.config["hidden_size"], output_dim=2, exp_name=self.hparams.config["exp_name"])
            self.nlvr2_classifier.apply(objectives.init_weights)


        fiber_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            for key in ['image_queue', 'text_queue', 'queue_ptr', 'queue_total', 'image_input_queue', 'text_input_queue', 'text_input_mask_queue']:
                if key in state_dict:
                    state_dict.pop(key)
            self.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, image_input, text_input, text_input_mask):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        image_input = concat_all_gather(image_input)
        text_input = concat_all_gather(text_input)
        text_input_mask = concat_all_gather(text_input_mask)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        ptr_total = int(self.queue_total)
        #assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.queue_size:
            self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
            self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
            self.image_input_queue[ptr:ptr+batch_size, :, :, :] = image_input
            self.text_input_queue[ptr:ptr+batch_size, :] = text_input
            self.text_input_mask_queue[ptr:ptr+batch_size, :] = text_input_mask
            ptr = (ptr + batch_size) % self.queue_size  # move pointer
        else:
            first_len = self.queue_size - ptr
            self.image_queue[:, ptr:] = image_feats[:first_len].T
            self.text_queue[:, ptr:] = text_feats[:first_len].T
            self.image_input_queue[ptr:, :, :, :] = image_input[:first_len]
            self.text_input_queue[ptr:, :] = text_input[:first_len]
            self.text_input_mask_queue[ptr:, :] = text_input_mask[:first_len]

            ptr = (ptr + batch_size) % self.queue_size  # move pointer
            self.image_queue[:, :ptr] = image_feats[first_len:].T
            self.text_queue[:, :ptr] = text_feats[first_len:].T
            self.image_input_queue[:ptr, :, :, :] = image_input[first_len:]
            self.text_input_queue[:ptr, :] = text_input[first_len:]
            self.text_input_mask_queue[:ptr, :] = text_input_mask[first_len:]
            
        ptr_total = ptr_total + batch_size

        self.queue_ptr[0] = ptr
        self.queue_total[0] = ptr_total

    def unnormalize(image_tensor, mean, std):
        # The mean and std have to be broadcastable to the tensor's shape
        mean = torch.as_tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device)
        std = torch.as_tensor(std, dtype=image_tensor.dtype, device=image_tensor.device)
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
    
        return image_tensor * std + mean
    
    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
        text_only=False,
        image_only=False,
    ):
        if not text_only:
            if img is None:
                if f"image_{image_token_type_idx - 1}" in batch:
                    imgkey = f"image_{image_token_type_idx - 1}"
                else:
                    imgkey = "image"
                img = batch["image"][0]

        if not image_only:
            do_mlm = "_mlm" if mask_text else ""
            text_ids = batch[f"text_ids{do_mlm}"]
            text_labels = batch[f"text_labels{do_mlm}"]
            text_masks = batch[f"text_masks"]
            text_data = batch[f"text"][0]

        # block attn
        if text_only:
            text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
            device = text_embeds.device
            input_shape = text_masks.size()
            extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
            for layer_i, layer in enumerate(self.text_transformer.encoder.layer):
                text_embeds = layer(text_embeds, extend_text_masks)[0]

            text_embeds = self.cross_modal_text_transform_itc(text_embeds)

            if self.itc_pooler:
                cls_feats_text = self.cross_modal_text_pooler_itc(text_embeds)
            else:
                cls_feats_text = text_embeds[:, 0]

            cls_feats_text = cls_feats_text / cls_feats_text.norm(dim=-1, keepdim=True)

            ret = {
                "text_feats": text_embeds,
                "image_feats": None,
                "cls_feats": cls_feats_text,
                "text_labels": text_labels,
                "text_ids": text_ids,
                "text_masks": text_masks,
                "image": None,
                "text": batch[f"text"][0],
            }

            return ret

        if image_only:
            image_embeds = self.vit_model.patch_embed(img)
            if self.vit_model.absolute_pos_embed is not None:
                image_embeds = image_embeds + self.vit_model.absolute_pos_embed
            image_embeds = self.vit_model.pos_drop(image_embeds)

            for layer_i, layer in enumerate(self.vit_model.layers):
                image_embeds = layer(image_embeds)
            image_embeds = self.vit_model.norm(image_embeds)
            image_embeds = self.cross_modal_image_transform_itc(image_embeds)
            image_feats = image_embeds

            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            if self.itc_pooler:
                cls_feats_image = self.cross_modal_image_pooler_itc(avg_image_feats)
            else:
                cls_feats_image = avg_image_feats[:, 0]

            cls_feats_image = cls_feats_image / cls_feats_image.norm(dim=-1, keepdim=True)

            ret = {
                "text_feats": None,
                "image_feats": image_embeds,
                "cls_feats": cls_feats_image,
                "text_labels": None,
                "text_ids": None,
                "text_masks": None,
                "image": None,
                "text": batch[f"text"][0],
                
            }

            return ret

        image_embeds = self.vit_model.patch_embed(img)
        if self.vit_model.absolute_pos_embed is not None:
            image_embeds = image_embeds + self.vit_model.absolute_pos_embed
        image_embeds = self.vit_model.pos_drop(image_embeds)
        for layer_i, layer in enumerate(self.vit_model.layers[:2]):
            image_embeds = layer(image_embeds)

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
        num_pre_text = self.num_text_layer - self.num_fuse_block
        for layer_i, layer in enumerate(self.text_transformer.encoder.layer[:num_pre_text]):
            text_embeds = layer(text_embeds, extend_text_masks)[0]

        num_pre_block = 8 + num_pre_text
        for blk_cnt, blk in enumerate(self.vit_model.layers[2].blocks):
            if blk_cnt < num_pre_block:
                image_embeds = blk(image_embeds)
            else:
                fuse_image_embeds = blk(image_embeds, text_embeds, extend_text_masks)
                text_embeds = self.text_transformer.encoder.layer[blk_cnt - 8](
                    text_embeds, extend_text_masks, encoder_hidden_states=(image_embeds)
                )[0]
                image_embeds = fuse_image_embeds

        if self.vit_model.layers[2].downsample is not None:
            image_embeds = self.vit_model.layers[2].downsample(image_embeds)

        for blk_cnt, blk in enumerate(self.vit_model.layers[3].blocks):
            fuse_image_embeds = blk(image_embeds, text_embeds, extend_text_masks)
            text_embeds = self.text_transformer.encoder.layer[blk_cnt + 10](
                text_embeds, extend_text_masks, encoder_hidden_states=(image_embeds), last_norm=(blk_cnt == 0)
            )[0]
            image_embeds = fuse_image_embeds

        if self.vit_model.layers[3].downsample is not None:
            image_embeds = self.vit_model.layers[3].downsample(image_embeds)

        text_embeds = self.cross_modal_text_transform(text_embeds)
        image_embeds = self.cross_modal_image_transform(image_embeds)

        cls_feats_text = self.cross_modal_text_pooler(text_embeds)
        avg_image_feats = self.avgpool(image_embeds.transpose(1, 2)).view(image_embeds.size(0), 1, -1)
        cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        ret = {
            "text_feats": text_embeds,
            "image_feats": image_embeds,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "image": img,
            "text": batch[f"text"][0],
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        fiber_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        fiber_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        fiber_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        fiber_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        fiber_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        outs = self.all_gather(outs)

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(self, outs, model_name)

        fiber_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        if "vqa" in self.exp_name:
            return torch.optim.AdamW(self.vqa_classifier.parameters(), lr=self.hparams.config["learning_rate"], eps=1e-8, betas=(0.9, 0.98))
        else:
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.config["learning_rate"], eps=1e-8, betas=(0.9, 0.98))