"""
main train code
"""

import os
import argparse

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.utils import get_option_dict_from_json, create_folder
from models.efficientnet import EfficientNet
from RMFER import RMFER

ALLOWED_MODELS = ["enet-b2"]
ALLOWED_OPTIMS = ["Adam"]

parser = argparse.ArgumentParser(description="hyper parameter json")
parser.add_argument(
    "--param",
    type=str,
    help="hyper paramter args path",
    default="./configs/Base/AffectNet7.json",
)
arg_path = parser.parse_args()


def train(args: dict):
    # assert model and optimizer
    model_name = args["learning_params"]["Base"]["model"]
    optim_name = args["learning_params"]["Base"]["optimizer"]
    assert model_name in ALLOWED_MODELS
    assert optim_name in ALLOWED_OPTIMS

    # default_root_dir
    default_root_dir = os.path.join(
        args["logging_params"]["default_root_dir"],
        args["exp_params"]["exp_mode"],
        args["logging_params"]["project"],
        args["logging_params"]["exp_name"],
    )
    create_folder(default_root_dir)

    # logger: WandB
    wandb_logger = WandbLogger(
        project=args["logging_params"]["project"],
        name=args["logging_params"]["exp_name"],
        save_dir=default_root_dir,
        log_model=False,
    )
    wandb_logger.log_hyperparams(params=args)

    # checkpointing
    checkpoint_path = os.path.join(default_root_dir, wandb_logger.experiment.id)
    create_folder(checkpoint_path)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_last=True,  # This will save a 'last.ckpt' file in the directory
        verbose=True,
        save_top_k=0,
    )

    # select model
    if model_name == "enet-b2":
        model = EfficientNet(
            emotions=args["exp_params"]["emotions"],
            self_masking=args["learning_params"]["Attention"]["self_masking"],
            scale=args["learning_params"]["Attention"]["scale"],
        )
    pl_module = RMFER(model=model, args=args)

    trainer = Trainer(
        # exp setting
        devices=args["exp_params"]["devices"],
        accelerator="gpu",
        strategy=args["exp_params"]["strategy"],
        deterministic=True,
        num_sanity_val_steps=2,
        # train setting
        max_epochs=args["exp_params"]["max_epochs"],
        reload_dataloaders_every_n_epochs=args["exp_params"][
            "reload_dataloader"
        ],
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        # log setting
        enable_model_summary=False,
        logger=wandb_logger,
        val_check_interval=args["logging_params"]["val_check_interval"],
        default_root_dir=default_root_dir,
    )
    trainer.fit(model=pl_module)


if __name__ == "__main__":
    input_args = get_option_dict_from_json(path=arg_path.param)

    # sets seeds for numpy, torch and python.random.
    seed_everything(input_args["exp_params"]["seed"], workers=True)
    torch.set_float32_matmul_precision("medium")

    # train by args
    train(input_args)
