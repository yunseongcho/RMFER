# pylint: disable=invalid-name
"""
RMFER lightning module
"""


import torch
from torch import nn
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS

from datasets.AffectNet import get_AffectNet_DataLoader


class RMFER_exp(pl.LightningModule):
    """_summary_

    Args:
        pl (_type_): _description_
    """

    def __init__(self, model, args) -> None:
        super().__init__()

        # self.automatic_optimization = False

        self.model = model
        self.args = args

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.args["data_params"]["main"]["dataset"] == "AffectNet":
            main_loader = get_AffectNet_DataLoader(args=self.args, isTrain=True)
        return main_loader

    def attention_step(self, inputs, labels):
        origin_outputs, att_outputs, _ = self.model(inputs)
        origin_loss = nn.CrossEntropyLoss()(origin_outputs, labels)
        att_loss = nn.CrossEntropyLoss()(att_outputs, labels)
        # ensemble of origin & attention
        loss_pretrain = origin_loss + (
            att_loss * self.args["learning_params"]["Attention"]["weight_att"]
        )
        return loss_pretrain

    def training_step(self, batch, _) -> STEP_OUTPUT:
        inputs, labels = batch
        return self.attention_step(inputs, labels)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.02)
