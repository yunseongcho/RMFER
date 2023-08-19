"""
etc functions
"""

import os
import json
from models.efficientnet import EfficientNet

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
