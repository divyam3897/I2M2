import argparse
import os
import pathlib
from typing import Dict, Optional, Tuple
import numpy as np

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from tqdm.auto import tqdm
from pathlib import Path

import sys
import inspect

from data.knee_data import KneeDataClassificationModule
from module import RSS
import fire
import itertools
import random


def get_data(args: argparse.ArgumentParser) -> pl.LightningDataModule:
    # get datamodule
    return KneeDataClassificationModule(args, label_type="knee")


def get_model(
    args: argparse.ArgumentParser, device: torch.device,
) -> pl.LightningModule:
    model = RSS(
            args,
            image_shape=[320, 320],
            kspace_shape=[640, 400],
            device=device,
        )
    return model

def train_model(
    args: argparse.Namespace,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    device: torch.device,
) -> pl.LightningModule:

    log_dir = (
        Path(args.log_dir)
        / args.data_type
        / args.data_space
        / str(args.n_seed)
        / str(args.lr)
        / str(args.weight_decay)
    )
    noise = [0]
   
    model_dir =  str(args.model_dir) + "/" + args.data_space + '/'  + str(args.n_seed) + '/' + str(args.lr) + '/' + str(args.weight_decay) + '/' + str(args.batch_size)

    if not os.path.isdir(str(log_dir)):
        try :
            os.makedirs(str(log_dir))
        except :
            print(f"Directory {str(log_dir)} already exists")
    if not os.path.isdir(str(model_dir)):
        try :
            os.makedirs(str(model_dir))
        except :
            print(f"Directory {str(model_dir)} already exists")

    csv_logger = CSVLogger(save_dir=log_dir, version=f"{args.n_seed}")
    wandb_logger = WandbLogger(name=f"{args.data_space}-{args.lr}-{args.weight_decay}", project=args.wandb_project, offline=args.wandb_offline)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model_checkpoint = ModelCheckpoint(monitor='val_auc_mean_max', dirpath=model_dir, filename="{epoch:02d}-{val_auc_mean_max:.2f}",save_top_k=1, mode='max')
    early_stop_callback = EarlyStopping(monitor='val_auc_mean_max', patience=5, mode='max', log_rank_zero_only=True)

    trainer: pl.Trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="ddp",#DDPStrategy(find_unused_parameters=False),
        max_epochs=args.n_epochs,
        logger=[wandb_logger, csv_logger],
        callbacks=[model_checkpoint, early_stop_callback, lr_monitor],
        sync_batchnorm=args.sync_batchnorm,
    )
    if args.skip_training :
        datamodule.setup()
        trainer.validate(model,datamodule.val_dataloader())
    else :
        trainer.fit(model, datamodule)

    return model


def test_model(
    args: argparse.Namespace,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    device: torch.device,
) -> pl.LightningModule:
    
    if args.ensemble:
      if not args.lp:
        models_list = list(itertools.combinations(range(1,6), 3))
        all_modalities = {'ktoi_w_mag': {'lr': 1e-05, 'weight_decay': 1e-3}, 'ktoi_w_phase': {'lr': 1e-4, 'weight_decay': 1e-1}, 'ktoi_w_magphase': {'lr': 1e-4, 'weight_decay': 1e-2}, 'ktoi_w_real': {'lr': 1e-5, 'weight_decay': 1e-1}, 'ktoi_w_imag':{'lr': 1e-4, 'weight_decay': 1e-1}}
        modality = ['ktoi_w_mag', 'ktoi_w_phase', 'ktoi_w_magphase']
        if modality[0] != modality[1]:
            from itertools import product
            models_list = random.sample(list(product(range(1,6), repeat=3)), k=10)
        for model_comb in models_list:
            model_path_1 = str(args.model_dir) + '/' + args.coil_type + "/" + modality[0] +  '/' + str(model_comb[0]) +  '/' + str(all_modalities[modality[0]]['lr']) + '/' + str(all_modalities[modality[0]]['weight_decay']) + '/' + str(args.batch_size)
            model_path_1 = model_path_1 + '/' + os.listdir(model_path_1)[0]
            print("Checkpoint file (modality 1): ", model_path_1)

            model_path_2 = str(args.model_dir) + '/' + args.coil_type + "/" + modality[1] +  '/' + str(model_comb[1]) +  '/' + str(all_modalities[modality[1]]['lr']) + '/' + str(all_modalities[modality[1]]['weight_decay']) + '/' + str(args.batch_size)
            model_path_2 = model_path_2 + '/' + os.listdir(model_path_2)[0]
            print("Checkpoint file (modality 2): ", model_path_2)

            model_path_3 = str(args.model_dir) + '/' + args.coil_type + "/" + modality[2] +  '/' + str(model_comb[2]) +  '/' + str(all_modalities[modality[2]]['lr']) + '/' + str(all_modalities[modality[2]]['weight_decay']) + '/' + str(args.batch_size)
            model_path_3 = model_path_3 + '/' + os.listdir(model_path_3)[0]
            print("Checkpoint file (modality 3): ", model_path_3)

            log_dir = (
                Path(args.log_dir)
                / args.coil_type
                / args.data_type
                / modality[0]
                / modality[1]
                / "ensemble"
                / str(model_comb[0])
                / str(model_comb[1])
                / str(model_comb[2])
            )
            model = Ensemble(args, model_path_1, model_path_2, model_path_3, device, return_features=False)
            model.eval()
            csv_logger = CSVLogger(save_dir=log_dir, name=f"test_noise-{args.noise_type}-{args.noise_percent}", version=f"{args.n_seed}")
            trainer = pl.Trainer(gpus=1 if str(device).startswith("cuda") else 0, logger=csv_logger)
            M_val = trainer.validate(model, datamodule.val_dataloader())  
            M = trainer.test(model, datamodule.test_dataloader())
      else:
        model_dir =  str(args.model_dir) + "/" + args.coil_type + '/' + args.data_space + '/'  + str(args.n_seed) + '/' + str(args.lr) + '/' + str(args.weight_decay) + '/' + str(args.batch_size)
        checkpoint_filename = os.listdir(model_dir)[0]
        model = DebiasedModule.load_from_checkpoint(model_dir + "/" + checkpoint_filename)

        log_dir = (
                Path(args.log_dir)
                / args.coil_type
                / args.data_type
                / args.data_space
        )
      csv_logger = CSVLogger(save_dir=log_dir, name=f"test_noise-{args.noise_percent}", version=f"{args.n_seed}")
      trainer = pl.Trainer(gpus=1 if str(device).startswith("cuda") else 0, logger=csv_logger)
      model.eval()
      M_val = trainer.validate(model, datamodule.val_dataloader())  
      M = trainer.test(model, datamodule.test_dataloader())
    else:
        model_dir = str(args.model_dir) + '/' + args.coil_type + '/' + args.data_space + '/'  + str(args.n_seed) + '/' + str(args.lr) + '/' + str(args.weight_decay) + '/' + str(args.batch_size)
        checkpoint_filename = os.listdir(model_dir)[0]
        print("Checkpoint file: ", model_dir, checkpoint_filename)
        bs = str(args.batch_size)
        log_dir = (
        Path(args.log_dir)
        / args.coil_type
        / args.data_type
        / args.data_space
        )
        csv_logger = CSVLogger(save_dir=log_dir, name=f"test_noise-{args.noise_type}-{args.noise_percent}", version=f"{args.n_seed}")
        model = RSS.load_from_checkpoint(model_dir + '/' + checkpoint_filename, norm=args.norm,  image_shape=[320, 320], kspace_shape=[640, 400],)
        trainer = pl.Trainer(gpus=1 if str(device).startswith("cuda") else 0, logger=csv_logger)

        model.eval()
        M_val = trainer.validate(model, datamodule.val_dataloader())  
        M = trainer.test(model, datamodule.test_dataloader())


def get_args():
    parser = argparse.ArgumentParser(description="Indirect MR Screener training")
    # logging parameters
    parser.add_argument("--model_dir", type=str, default="./trained_models_wide/knee")
    parser.add_argument("--log_dir", type=str, default="./trained_logs/knee")

    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--dev_mode", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="fastmri_runs")
    # data parameterssq
    parser.add_argument(
        "--data_type", type=str, default="knee",
    )
    parser.add_argument(
        "--data_space", type=str, default="ktoi_w_realimag")
    parser.add_argument(
        "--task", type=str, default="classification",
    )
    parser.add_argument("--image_shape", type=int, default=[320, 320], nargs=2, required=False)
    parser.add_argument("--image_type", type=str, default='orig', required=False, choices=["orig"])
    parser.add_argument("--split_csv_file", type=str, default='./splits/metadata_knee_sc.csv', required=False)

    parser.add_argument("--loss_fn_weights_filename", default='./data/loss_fn_weights_knee_tr.p')

    parser.add_argument(
        "--model_type",
        type=str,
        default="preact_resne18",
    )
    parser.add_argument(
        "--model_type_class",
        type=str,
        default="multidimensional",
    )

    # training parameters
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--n_seed", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--drop_prob", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--sweep_step", type=int)
    parser.add_argument('--debug',  action='store_true')
    parser.add_argument('--downsample', action='store_true')
    parser.add_argument('--wandb_offline', action='store_true')
    parser.add_argument('--shared_conv', action='store_true')
    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--sync_batchnorm', action="store_true")
    parser.add_argument('--fc_layer_dim', type=int, default=100)
    parser.add_argument("--norm", type=str, default="layer", required=False, choices=["group", "batch", "layer", "instance"],)

    args, unkown = parser.parse_known_args()
    
    return args


def retreve_config(args, sweep_step=None):
    if sweep_step is None :
        return args
    grid = {
        "n_seed": [1,2,3,4,5],
    }

    grid_setups = list(
        dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())
    )
    step_grid = grid_setups[sweep_step - 1]  # slurm var will start from 1

    # automatically choose the device based on the given node
    if torch.cuda.device_count() > 0:
        expr_device = "cuda"
    else:
        expr_device = "cpu"
    
    args.sweep_step = sweep_step
    args.n_seed = step_grid['n_seed']

    return args


def run_experiment(args):
    
    print(args, flush=True)
    if torch.cuda.is_available():
        print("Found CUDA device, running job on GPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datamodule = get_data(args)
    model = get_model(args, device)
    if args.mode == "train":
        model = train_model(args=args, model=model, datamodule=datamodule, device=device,)
    else:
        datamodule.setup()
        test_model(args=args, model=model, datamodule=datamodule, device=device,)

def main(sweep_step=None):
    args = get_args()
    config = retreve_config(args, sweep_step)
    run_experiment(config)


if __name__ == "__main__":
     fire.Fire(main)
