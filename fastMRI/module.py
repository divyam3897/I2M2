from collections import OrderedDict
import os
from typing import Tuple, Optional
from torch.serialization import validate_cuda_device
from tqdm.auto import tqdm
import numpy as np
import copy
from copy import deepcopy
import pytorch_lightning as pl

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from joblib import dump, load

from metrics.classification_metrics import (
    compute_accuracy,
    get_operating_point,
    evaluate_classifier,
    get_bootstrap_estimates,
)
from torch.utils.data import DataLoader


from models.preactresnet_knee import PreActResNet18

class RSS(pl.LightningModule):
    def __init__(
        self,
        args,
        kspace_shape: Tuple[int, int],
        image_shape: Tuple[int, int],
        device: torch.device,
        coil_type: str = "sc",
        label_names: list=["abnormal", "mtear", "acl", "cartilage"],
        num_labels = 4,
        n_bootstrap_samples: int = 50,
        sequences: Optional[Tuple[str, str, str]] = ["t2", "b50"],
        return_features: str = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # data and task type
        self.data_type = args.data_type
        self.image_shape = image_shape
        self.mode = args.mode
        if args.loss_fn_weights_filename is not None:
            self.loss_fn_weights = load(args.loss_fn_weights_filename)

        # model type and parameters
        self.model_type = args.model_type
        self.drop_prob = args.drop_prob
        self.label_names = label_names
        self.num_labels = num_labels

        # optimizer parameters
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.kspace_shape = kspace_shape
        self.model_type_class = args.model_type_class

        if args.loss_fn_weights_filename is not None:
            self.criterion_abnormal = nn.CrossEntropyLoss(weight=self.loss_fn_weights[0].float())        
            self.criterion_mtear = nn.CrossEntropyLoss(weight=self.loss_fn_weights[1].float())        
            self.criterion_acl = nn.CrossEntropyLoss(weight=self.loss_fn_weights[2].float())        
            self.criterion_cartilage = nn.CrossEntropyLoss(weight=self.loss_fn_weights[3].float())     

        self.sequences = sequences
        self.data_space = args.data_space
        self.return_features = return_features
        self.coil_type = coil_type
        self.downsample = False
        
        self.debug = args.debug
        
        # get model depending on data and model type
        if self.data_space == "ktoi_w_rss":
            self.model = PreActResNet18(image_shape=self.image_shape, drop_prob=self.drop_prob, data_space="ktoi_w_rss", return_features=return_features)
            
        if self.model_type_class == "multimodal":
            self.model_cat = PreActResNet18(image_shape=self.image_shape, drop_prob=self.drop_prob, data_space="ktoi_w_magphase", return_features=return_features)
            self.model_real = PreActResNet18(image_shape=self.image_shape, drop_prob=self.drop_prob, data_space="ktoi_w_phase", return_features=return_features)
            self.model_imag = PreActResNet18(image_shape=self.image_shape, drop_prob=self.drop_prob, data_space="ktoi_w_mag", return_features=return_features)
        else:
            self.model_cat = PreActResNet18(image_shape=self.image_shape, drop_prob=self.drop_prob, data_space=self.data_space, return_features=return_features)
            
        self.val_operating_point = None
        self.n_bootstrap_samples = n_bootstrap_samples
        
        # declare dictionaries to keep track of the best metrics
        self.val_auc_max = {}
        for name in self.label_names :
            self.val_auc_max[name] = 0.0
        self.val_auc_max['mean'] = 0.0 # add extra label to keep track of best mean auc

        self.val_bac_max = {} # add extra label to keep track of best mean bac
        for name in self.label_names :
            self.val_bac_max[name] = 0.0
        self.val_bac_max['mean'] = 0.0 # add extra label to keep track of best mean bac


    def loss_fn_abnormal(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion_abnormal(preds, labels)

    def loss_fn_mtear(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion_mtear(preds, labels)

    def loss_fn_acl(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion_acl(preds, labels)

    def loss_fn_cartilage(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion_cartilage(preds, labels)
    
    def forward(self, batch, data_space=None):
        if self.data_space == "ktoi_w_rss" :
            image = batch.recon_rss.cuda()
            return self.model(image.unsqueeze(1))
        elif self.coil_type == "sc" :
            kspace = batch.sc_kspace
            kspace = kspace.cuda()
            outputs_cat = self.model_cat(kspace.unsqueeze(1), data_space)
            if self.model_type_class == "multimodal":
                outputs_real = self.model_real(kspace.unsqueeze(1))
                outputs_imag = self.model_imag(kspace.unsqueeze(1))

                for i in range(0, self.num_labels):

                    out_num = (torch.log_softmax(outputs_real[i], dim=-1)) + \
                              (torch.log_softmax(outputs_imag[i], dim=-1)) + \
                              (torch.log_softmax(outputs_cat[i], dim=-1))

                    out_den = torch.logsumexp(out_num, dim=-1)
                    outputs_cat[i] = out_num  - out_den.unsqueeze(1)

            return outputs_cat
        
       
    def loss_fn(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(preds, labels)

    def compute_loss_and_metrics(self, preds, labels, label_names):
        if self.skip_training :
            return None
        assert len(label_names) == self.num_labels
            
        pred_out, label_out = [], [] # To store preds and labels for each label
        acc_per_label = []

        loss = None
        for i in range(0, self.num_labels):
            
            curr_loss = self.loss_fn(preds=preds[i], labels=labels[:,i])
            if loss is None:
                loss = curr_loss
            else:
                loss += curr_loss
                
            acc = compute_accuracy(preds[i], labels[:, i])
            acc_per_label.append(acc)
            
            self.log(label_names[i], acc, prog_bar=True, batch_size=preds[i].shape)
            
        return loss

    def training_step(self, batch, batch_idx):
        labels = batch.label.long()
        
        # get predictions
        preds = self.forward(batch=batch)
        labels = labels.long()
        labels_abnormal = labels[:, 0]
        labels_mtear = labels[:, 1]
        labels_acl = labels[:, 2]
        labels_cartilage = labels[:, 3]

        preds_abnormal, preds_mtear, preds_acl, preds_cartilage = preds
        loss_abnormal = self.loss_fn_abnormal(preds_abnormal, labels_abnormal)
        loss_mtear = self.loss_fn_mtear(preds_mtear, labels_mtear)
        loss_acl = self.loss_fn_acl(preds_acl, labels_acl)
        loss_cartilage = self.loss_fn_cartilage(preds_cartilage, labels_cartilage)
        loss = loss_abnormal + loss_mtear + loss_acl + loss_cartilage
                            
        acc_abnormal = compute_accuracy(preds_abnormal.max(1)[1], labels_abnormal)
        acc_mtear = compute_accuracy(preds_mtear.max(1)[1], labels_mtear)
        acc_acl = compute_accuracy(preds_acl.max(1)[1], labels_acl)
        acc_cartilage = compute_accuracy(preds_cartilage.max(1)[1], labels_cartilage)

        self.log("train_abnormal_acc", acc_abnormal, prog_bar=True, on_step=True, on_epoch=True, batch_size=preds_abnormal.shape[0])
        self.log("train_mtear_acc", acc_mtear, prog_bar=True, on_step=True, on_epoch=True, batch_size=preds_mtear.shape[0])
        self.log("train_acl_acc", acc_acl, prog_bar=True, on_step=True, on_epoch=True, batch_size=preds_acl.shape[0])
        self.log("train_cartilage", acc_cartilage, prog_bar=True, on_step=True, on_epoch=True, batch_size=preds_cartilage.shape[0])
        
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        labels = batch.label.long()
        # get predictions
        preds = self.forward(batch=batch)
        labels_abnormal = labels[:, 0]
        labels_mtear = labels[:, 1]
        labels_acl = labels[:, 2]
        labels_cartilage = labels[:, 3]

   
        return {"batch_idx": batch_idx,
                "batch_size": labels.shape[0],
                "labels": labels,
                "preds": preds,
                }

    def collate_results(self, logs: Tuple) -> Tuple:
        loss = []
        loss_list = []
        n_sample_points = 0

        labels_abnormal, labels_mtear, labels_acl, labels_cartilage = [], [], [], []
        preds_abnormal, preds_mtear, preds_acl, preds_cartilage = [], [], [], []
            
        for log_t in logs:
            n_sample_points += log_t["batch_size"]
            preds_t, labels_t = log_t["preds"], log_t["labels"]

            labels_abnormal.append(labels_t[:, 0])
            labels_mtear.append(labels_t[:, 1])
            labels_acl.append(labels_t[:, 2])
            labels_cartilage.append(labels_t[:, 3])

            preds_abnormal.append(preds_t[0])
            preds_mtear.append(preds_t[1])
            preds_acl.append(preds_t[2])
            preds_cartilage.append(preds_t[3])

        labels_abnormal = torch.cat(labels_abnormal, dim=0)
        labels_mtear = torch.cat(labels_mtear, dim=0)
        labels_acl = torch.cat(labels_acl, dim=0)
        labels_cartilage = torch.cat(labels_cartilage, dim=0)

        preds_abnormal = torch.cat(preds_abnormal, dim=0)
        preds_mtear = torch.cat(preds_mtear, dim=0)
        preds_acl = torch.cat(preds_acl, dim=0)
        preds_cartilage = torch.cat(preds_cartilage, dim=0)


        labels = [labels_abnormal, labels_mtear, labels_acl, labels_cartilage]
        preds = [preds_abnormal, preds_mtear, preds_acl, preds_cartilage]

        return {"labels": labels,
                "preds": preds
            }

    def validation_epoch_end(self, val_logs):
        collate_output = self.collate_results(val_logs)
        preds = self.all_gather(collate_output["preds"])
        labels = self.all_gather(collate_output["labels"])
        
        if len(preds[0].shape) > 2:
            preds = [pred.reshape(pred.shape[0]*pred.shape[1], pred.shape[2]) for pred in preds]
            labels = [label.reshape(label.shape[0]*label.shape[1]) for label in labels]

        avg_auc = 0.0
        avg_bac = 0.0
        batch_size = labels[0].shape[0]
        self.log("lr", self.lr, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("weight_decay", self.weight_decay, prog_bar=True, batch_size=batch_size, sync_dist=True)

        labels_abnormal, labels_mtear, labels_acl, labels_cartilage = labels
        preds_abnormal, preds_mtear, preds_acl, preds_cartilage = preds

        eval_metrics = {}
        keys = ["abnormal", "mtear", "acl", "cartilage"]
        for i, key in enumerate(keys):
            eval_metrics[key] = evaluate_classifier(preds[i], labels[i])

        self.val_operating_point = {}
            
        for key in keys:
            key_score = eval_metrics[key]["auc"]
            key_acc = eval_metrics[key]["balanced_accuracy"]
            self.log(f"val_auc_{key}", key_score, prog_bar=True, batch_size=batch_size,sync_dist=True)
            self.log(
                    f"val_bac_{key}",
                    eval_metrics[key]["balanced_accuracy"],
                    prog_bar=True,
                    batch_size=batch_size,
                    sync_dist=True,
            )
            avg_auc += key_score / len(keys)
            avg_bac += key_acc / len(keys)

            self.val_operating_point[key] = eval_metrics[key]["operating_point"]
        
        if avg_auc > self.val_auc_max["mean"] :
            self.val_auc_max["mean"] = avg_auc
            for key in keys :
                key_score = eval_metrics[key]["auc"]
                self.val_auc_max[key] = key_score

        if avg_bac > self.val_bac_max["mean"] :
            self.val_bac_max["mean"] = avg_bac
            for key in keys :
                key_acc = eval_metrics[key]["balanced_accuracy"]
                self.val_bac_max[key] = key_acc

        for k,v in self.val_auc_max.items() :
            self.log(f"val_auc_{k}_max", v, prog_bar=True, batch_size=batch_size,sync_dist=True)
        for k,v in self.val_bac_max.items() :
            self.log(f"val_bac_{k}_max", v, prog_bar=True, batch_size=batch_size,sync_dist=True)

        self.log(f"val_auc_mean", avg_auc, prog_bar=True, batch_size=batch_size,sync_dist=True)

    def test_step(self, batch, batch_idx):
        labels = batch.label.long()
        preds = self.forward(batch=batch)

        labels_abnormal = labels[:, 0]
        labels_mtear = labels[:, 1]
        labels_acl = labels[:, 2]
        labels_cartilage = labels[:, 3]


        return {"batch_idx": batch_idx,
                "batch_size": labels.shape[0],
                "labels": labels,
                "preds": preds,
                }

    def test_epoch_end(self, test_logs):
        collate_output = self.collate_results(test_logs)
        preds = self.all_gather(collate_output["preds"])
        labels = self.all_gather(collate_output["labels"])
        batch_size = preds[0].shape[0]

        if self.data_type == "knee":
            labels_abnormal, labels_mtear, labels_acl, labels_cartilage = labels
            preds_abnormal, preds_mtear, preds_acl, preds_cartilage = preds

            probs = torch.bincount(labels_acl) / len(labels_acl)
            probs = probs[probs > 0]
           
            eval_metrics = {}
            keys = ["abnormal", "mtear", "acl", "cartilage"]

            for i, key in enumerate(keys):
                eval_metrics[key] = evaluate_classifier(preds[i], labels[i])

            avg_auc = 0.0
            test_operating_point = {}
            keys = ["abnormal", "mtear", "acl", "cartilage"]
            loss = 0
            prefix = f"test"
            for key in ["abnormal", "mtear", "acl", "cartilage"]:
                for metric in ["auc",
                                "sensitivity",
                                "specificity",
                                "balanced_accuracy",
                                "operating_point"]:
                    key_score = eval_metrics[key][metric]
                    self.log(
                        f"{prefix}_{key}_{metric}",
                        eval_metrics[key][metric],
                        prog_bar=True,
                        batch_size=batch_size,
                        rank_zero_only=True
                     )
            
        return loss, eval_metrics

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        return [optimizer]
