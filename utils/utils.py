"""
etc functions
"""

import os
import json
from models.efficientnet import EfficientNet
from lightning.pytorch.loggers import WandbLogger

ALLOWED_MODELS = ["enet-b2"]
ALLOWED_OPTIMS = ["Adam"]


def create_folder(path: str):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: Creating directory. {path}")


def get_option_dict_from_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)

    return result


def save_option_json_from_dict(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def select_model(args):
    model_name = args["learning_params"]["Base"]["model"]
    assert model_name in ALLOWED_MODELS
    if model_name == "enet-b2":
        model = EfficientNet(
            emotions=args["exp_params"]["emotions"],
            self_masking=args["learning_params"]["Attention"]["self_masking"],
            scale=args["learning_params"]["Attention"]["scale"],
        )

    return model


def get_wandb_logger(args, default_root_dir):
    if not args["logging_params"]["wandb_id"]:
        wandb_logger = WandbLogger(
            project=args["logging_params"]["project"],
            name=args["logging_params"]["exp_name"],
            save_dir=default_root_dir,
            log_model=False,
        )
        args["logging_params"]["wandb_id"] = wandb_logger.experiment.id
        wandb_logger.log_hyperparams(params=args)
        is_resume = False
    else:
        wandb_logger = WandbLogger(
            project=args["logging_params"]["project"],
            name=args["logging_params"]["exp_name"],
            save_dir=default_root_dir,
            log_model=False,
            id=args["logging_params"]["wandb_id"],
        )
        is_resume = True
    return wandb_logger, is_resume
