"""
main train code
"""

import argparse

from lightning.pytorch import Trainer, seed_everything

from utils.utils import get_option_dict_from_json
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
    model_name = args["learning_params"]["Base"]["model"]
    optim_name = args["learning_params"]["Base"]["optimizer"]
    assert model_name in ALLOWED_MODELS
    assert optim_name in ALLOWED_OPTIMS

    if model_name == "enet-b2":
        model = EfficientNet(
            emotions=args["exp_params"]["emotions"],
            self_masking=args["learning_params"]["Attention"]["self_masking"],
            scale=args["learning_params"]["Attention"]["scale"],
        )
    pl_module = RMFER(model=model, args=args)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        deterministic=True,
        max_epochs=2,
        reload_dataloaders_every_n_epochs=True,
    )
    trainer.fit(model=pl_module)


if __name__ == "__main__":
    input_args = get_option_dict_from_json(path=arg_path.param)

    # sets seeds for numpy, torch and python.random.
    seed_everything(input_args["exp_params"]["seed"], workers=True)

    # train by args
    train(input_args)
