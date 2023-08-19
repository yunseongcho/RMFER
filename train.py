"""
main train code
"""

import os
import argparse

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.utils import (
    get_option_dict_from_json,
    save_option_json_from_dict,
    create_folder,
    select_model,
)
from lightning_module_rmfer import Experiment


parser = argparse.ArgumentParser(description="hyper parameter json")
parser.add_argument(
    "--param",
    type=str,
    help="hyper parameter args path",
    default="./configs/Base/AffectNet7.json",
)
arg_path = parser.parse_args()


def train(args: dict):
    # make default_root_dir
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

    # check this experiment is resumed
    if not args["logging_params"]["wandb_id"]:
        args["logging_params"]["wandb_id"] = wandb_logger.experiment.id
        is_resume = False
        save_option_json_from_dict(f"{checkpoint_path}/args.json", args)
    else:
        wandb_logger.experiment.id = args["logging_params"]["wandb_id"]
        is_resume = True

    # select model
    model = select_model(args)

    # init lightning module
    pl_module = Experiment(model=model, args=args)

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
    trainer.fit(
        model=pl_module, ckpt_path=checkpoint_path if is_resume else None
    )


if __name__ == "__main__":
    input_args = get_option_dict_from_json(path=arg_path.param)

    # sets seeds for numpy, torch and python.random.
    seed_everything(input_args["exp_params"]["seed"], workers=True)
    torch.set_float32_matmul_precision("medium")

    # train by args
    train(input_args)
