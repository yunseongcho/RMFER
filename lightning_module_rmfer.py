# pylint: disable=invalid-name
"""
RMFER lightning module
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.utilities import CombinedLoader
from torchmetrics import Accuracy  # , ConfusionMatrix

from optimizer.SAM import SAMOptimizer
from datasets.AffectNet import get_AffectNet_DataLoader

ALLOWED_OPTIMS = ["Adam", "SAM"]


class Experiment(pl.LightningModule):
    """
    RMFER experiment
    """

    def __init__(self, model, args: dict) -> None:
        super().__init__()

        self.automatic_optimization = False

        self.model = model
        self.args = args

        # Best Measure initialize
        self.accuracy = Accuracy(
            task="multiclass", num_classes=args["exp_params"]["emotions"]
        )

    def train_dataloader(self) -> DataLoader | CombinedLoader:
        if self.args["data_params"]["main"]["dataset"] == "AffectNet":
            main_loader = get_AffectNet_DataLoader(args=self.args, isTrain=True)
        return main_loader

    def attention_step(self, inputs, labels):
        origin_outputs, att_outputs, _ = self.model.forward_with_att(inputs)
        origin_loss = nn.CrossEntropyLoss()(origin_outputs, labels)
        att_loss = nn.CrossEntropyLoss()(att_outputs, labels)
        # ensemble of origin & attention
        loss_pretrain = origin_loss + (
            att_loss * self.args["learning_params"]["Attention"]["weight_att"]
        )
        return loss_pretrain

    def training_step(self, batch, _):
        # data
        inputs, labels = batch

        # optimizer define
        opt = self.optimizers()

        ## SAM
        # calculate loss & first step
        loss_pretrain = self.attention_step(inputs, labels)
        self.manual_backward(loss_pretrain)
        opt.first_step(zero_grad=True)

        # calculate loss & second step
        loss_pretrain = self.attention_step(inputs, labels)
        self.manual_backward(loss_pretrain)
        opt.second_step(zero_grad=True)

        # log loss
        self.log("train_loss", loss_pretrain)

    def val_dataloader(self) -> CombinedLoader:
        if self.args["data_params"]["main"]["dataset"] == "AffectNet":
            train_loader = get_AffectNet_DataLoader(
                args=self.args, isTrain=True
            )
            val_loader = get_AffectNet_DataLoader(args=self.args, isTrain=False)
        iterables = {"train": train_loader, "val": val_loader}
        combined_loader = CombinedLoader(iterables, mode="min_size")
        return combined_loader

    def validation_step(self, batch, _):
        # train_inputs, train_labels = batch["train"]
        val_inputs, val_labels = batch["val"]

        # train_preds = self.model(train_inputs)
        val_preds = self.model(val_inputs)

        self.accuracy(val_preds, val_labels)

    def on_validation_end(self) -> None:
        val_acc = self.accuracy.compute()
        self.logger.experiment.log({"val_acc": val_acc})
        self.accuracy.reset()

    def configure_optimizers(self):
        optimizer = self._select_optim()
        return optimizer

    def _select_optim(self):
        optim_name = self.args["learning_params"]["Base"]["optimizer"]
        assert optim_name in ALLOWED_OPTIMS

        if optim_name == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args["learning_params"]["Base"]["learning_rate"],
            )
        elif optim_name == "SAM":
            optimizer = SAMOptimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                torch.optim.Adam,
                lr=self.args["learning_params"]["Base"]["learning_rate"],
            )
        return optimizer
