import os
import copy
import pytorch_lightning as pl
import os

os.environ["NCCL_DEBUG"] = "INFO"

from fiber.config import ex
from fiber.modules import FIBERTransformerSS
from fiber.datamodules.multitask_datamodule import MTDataModule

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    model = FIBERTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
        dirpath=_config["log_dir"], 
        filename="best_{epoch:02d}-{val/the_metric:.2f}",
        auto_insert_metric_name=True
    )
    csv_logger = pl.loggers.CSVLogger(
        _config["log_dir"],
        name="",
        version=_config["seed"]
    )
    if _config["test_only"]:
        loggers = [csv_logger]
    else:
        wandb_logger = pl.loggers.WandbLogger(name=f"{_config['exp_name']}-{_config['learning_rate']}", project='multimodal_vqa')
        loggers = [csv_logger, wandb_logger] 

    model.freeze()
    task = _config["exp_name"]
    if "vqa" in task:
        model.vqa_classifier.requires_grad_(True)
    else:
        raise ValueError

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = _config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"])

    grad_steps = max(_config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]), 1)

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="gpu",
        strategy="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"],
        callbacks=callbacks,
        logger=loggers,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
