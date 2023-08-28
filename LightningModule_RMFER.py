# pylint: disable=invalid-name
"""
RMFER lightning module
"""


import lightning.pytorch as pl
import numpy as np
import torch
import wandb
from lightning.pytorch.utilities import CombinedLoader

from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, ConfusionMatrix

from utils.experiments import (
    update_accuracy_confusion_matrix,
    update_attention_matrix,
    get_confusion_matrix_plot,
    get_attention_matrix_plot,
    img_from_buffer,
)
from datasets.AffectNet import get_AffectNet_DataLoader
from optimizer.SAM import SAMOptimizer

ALLOWED_OPTIMS = ["Adam", "SAM"]


class Experiment(pl.LightningModule):
    """
    RMFER experiment
    """

    def __init__(self, model, args: dict, default_root_dir: str) -> None:
        super().__init__()

        self.automatic_optimization = False

        self.model = model
        self.args = args
        self.default_root_dir = default_root_dir

        # expression label: num -> string dictionary
        self.label2exp_dict = {
            0: "Neutral",
            1: "Happiness",
            2: "Sadness",
            3: "Surprise",
            4: "Fear",
            5: "Disgust",
            6: "Anger",
            7: "Contempt",
        }

        # expression label list
        self.emotions = self.args["exp_params"]["emotions"]
        if self.emotions == 7:
            self.expression_labels = [
                "Neutral",
                "Happiness",
                "Sadness",
                "Surprise",
                "Fear",
                "Disgust",
                "Anger",
            ]
        elif self.emotions == 8:
            self.expression_labels = [
                "Neutral",
                "Happiness",
                "Sadness",
                "Surprise",
                "Fear",
                "Disgust",
                "Anger",
                "Contempt",
            ]

        # key initialize
        self.data_keys = ["train", "valid"]
        self.inference_keys = ["origin", "attention"]
        self.attention_keys = ["value", "count"]

        # Important Measure initialize
        self.init_Best_Measure()
        self.init_attention_matrix()
        self.init_accuracy_confusion_matrix()

    # -------------------------------------------------------------------------------------#
    # for initialization of measure
    def init_Best_Measure(self):
        # Best Performance for validation
        self.best_accuracy = {}
        self.best_similarity = {}
        self.best_similarity["equal"] = 0
        self.best_similarity["different"] = 100

        for measure_type in ["overall", "average"]:
            # overall accuracy or average accuracy
            self.best_accuracy[measure_type] = {}

            # origin or attention
            for inference_type in self.inference_keys:
                self.best_accuracy[measure_type][inference_type] = 0

    def init_accuracy_confusion_matrix(self):
        self.accuracy = {}
        self.confusion_matrix = {}
        self.validation_step_outputs = []

        # initialize Accuracies & ConfMat
        for data_type in self.data_keys:
            # make dict by data type
            self.accuracy[data_type] = {}
            self.confusion_matrix[data_type] = {}

            # make measure by inference type
            for inference_type in self.inference_keys:
                self.accuracy[data_type][inference_type] = Accuracy(
                    task="multiclass", num_classes=self.emotions
                ).cuda()
                # Accuracy(compute_on_step=False, dist_sync_on_step=True).cuda()
                self.confusion_matrix[data_type][
                    inference_type
                ] = ConfusionMatrix(
                    task="multiclass", num_classes=self.emotions
                ).cuda()
                # , compute_on_step=False, dist_sync_on_step=True).cuda()

    # Attention Measure init or reset
    def init_attention_matrix(self):
        # for measure cosine sim
        self.features = {}
        self.labels = {}
        self.attention_matrix = {}

        # data type: train or valid
        for data_type in self.data_keys:
            self.features[data_type] = []
            self.labels[data_type] = []
            self.attention_matrix[data_type] = {}

            # count or value
            for value_count in self.attention_keys:
                self.attention_matrix[data_type][value_count] = torch.zeros(
                    self.emotions, self.emotions
                ).cuda()

    # Accuracy & Confusion Matrix Measure reset
    def reset_accuracy_confusion_matrix(self):
        self.validation_step_outputs = []

        ### initialize Accuracies & ConfMat
        for data_type in self.data_keys:
            for inference_type in self.inference_keys:
                self.accuracy[data_type][inference_type].reset()
                self.confusion_matrix[data_type][inference_type].reset()

    # -------------------------------------------------------------------------------------#
    # for Training
    def train_dataloader(self) -> DataLoader | CombinedLoader:
        if self.args["data_params"]["main"]["dataset"] == "AffectNet":
            main_loader = get_AffectNet_DataLoader(args=self.args, isTrain=True)
        return main_loader

    def attention_step(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        pre-training step

        Args:
            inputs (torch.Tensor): _description_
            labels (torch.Tensor): _description_

        Returns:
            torch.Tensor: Loss_pretrain
        """
        (
            origin_outputs,
            attention_outputs,
            _,
            _,
        ) = self.model(inputs)
        origin_loss = nn.CrossEntropyLoss()(origin_outputs, labels)
        attention_loss = nn.CrossEntropyLoss()(attention_outputs, labels)

        # ensemble of origin & attention
        loss_pretrain = origin_loss + (
            attention_loss
            * self.args["learning_params"]["Attention"]["weight_att"]
        )

        self.log("origin_loss_in_epoch", origin_loss)
        self.log("attention_loss_in_epoch", attention_loss)
        self.log(
            "weighted_attention_loss_in_epoch",
            attention_loss
            * self.args["learning_params"]["Attention"]["weight_att"],
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
        self.log("total_loss", loss_pretrain)

        # return loss_pretrain

    # -------------------------------------------------------------------------------------#
    # for Validation
    def val_dataloader(self) -> CombinedLoader:
        if self.args["data_params"]["main"]["dataset"] == "AffectNet":
            train_loader = get_AffectNet_DataLoader(
                args=self.args, isTrain=True
            )
            valid_loader = get_AffectNet_DataLoader(
                args=self.args, isTrain=False
            )
        iterables = {"train": train_loader, "valid": valid_loader}
        combined_loader = CombinedLoader(iterables, mode="min_size")
        return combined_loader

    def validation_outputs(self, inputs, labels):
        (
            origin_outputs,
            attention_outputs,
            A,
            origin_features,
        ) = self.model(inputs)

        origin_loss = nn.CrossEntropyLoss()(origin_outputs, labels)
        attention_loss = nn.CrossEntropyLoss()(attention_outputs, labels)
        loss_dic = {"origin": origin_loss, "attention": attention_loss}

        return origin_outputs, attention_outputs, A, loss_dic, origin_features

    def validation_step_by_data_type(self, inputs, labels, data_type: str):
        # origin
        (
            origin_outputs,
            attention_outputs,
            A,
            loss_dic,
            origin_features,
        ) = self.validation_outputs(inputs, labels)

        # update Acc ConfMat
        accuracy = self.accuracy[data_type]
        confusion_matrix = self.confusion_matrix[data_type]

        self.features[data_type].append(origin_features)
        self.labels[data_type].append(labels)

        # origin
        update_accuracy_confusion_matrix(
            outputs=origin_outputs,
            labels=labels,
            accuracy=accuracy["origin"],
            confusion_matrix=confusion_matrix["origin"],
        )
        # attention
        update_accuracy_confusion_matrix(
            outputs=attention_outputs,
            labels=labels,
            accuracy=accuracy["attention"],
            confusion_matrix=confusion_matrix["attention"],
        )

        # update Attention Matrix
        attention_matrix = self.attention_matrix[data_type]
        update_attention_matrix(
            batch_attention_matrix=A,
            value_attention_matrix=attention_matrix["value"],
            count_attention_matrix=attention_matrix["count"],
            labels=labels,
        )

        # loss 계산
        return loss_dic

    def validation_step(self, batch, _):
        train_inputs, train_labels = batch["train"]
        valid_inputs, valid_labels = batch["valid"]

        # ensemble ver
        loss = {}
        loss["train"] = self.validation_step_by_data_type(
            inputs=train_inputs, labels=train_labels, data_type="train"
        )
        loss["valid"] = self.validation_step_by_data_type(
            inputs=valid_inputs, labels=valid_labels, data_type="valid"
        )

        self.validation_step_outputs.append(loss)

    # -------------------------------------------------------------------------------------#
    # calculate specific expression measure
    # use in validation epoch end
    def specific_expression_accuracy_log(
        self,
        data_type: str,
        inference_type: str,
        confusion_matrix: ConfusionMatrix,
    ):
        n_emotion = self.emotions
        recalls = np.zeros(n_emotion)
        for emotion in range(n_emotion):
            # recall 계산 try
            recall = (
                confusion_matrix[emotion][emotion].item()
                / confusion_matrix[emotion].sum().item()
            )
            recalls[emotion] = recall

        ## Logging
        for emotion in range(n_emotion):
            expression_name = self.label2exp_dict[emotion]
            txt = f"{data_type}_{inference_type}_{expression_name}_recall"
            self.logger.experiment.log({txt: recalls[emotion]})

        return recalls.mean()

    # log Confusion Matrix image
    def log_confusion_matrix(
        self,
        confusion_matrix: ConfusionMatrix,
        inference_type: str,
        measure: str,
    ):
        confusion_matrix = confusion_matrix.cpu()
        confusion_image = img_from_buffer(
            get_confusion_matrix_plot(
                confusion_matrix.numpy(), self.expression_labels
            )
        )
        wandb.log(
            {
                f"[Best] valid_{inference_type}_{measure}_confusion_matrix": [
                    wandb.Image(confusion_image, caption="Confusion Matrix")
                ]
            }
        )

    # log Attention Matrix image
    def log_attention_matrix(self, attention_matrix, txt: str):
        AttImage = img_from_buffer(
            get_attention_matrix_plot(attention_matrix, self.expression_labels)
        )
        wandb.log(
            {
                f"{txt}_attention_matrix": [
                    wandb.Image(AttImage, caption="Attention Matrix")
                ]
            }
        )

    def val_end_losses(self, losses):
        # total loss initialize, total_loss[data_type][inference_type]
        total_loss = {}
        for data_type in self.data_keys:
            total_loss[data_type] = {}
            for inference_type in self.inference_keys:
                total_loss[data_type][inference_type] = 0.0

        # loss summation
        for data_type in self.data_keys:
            for inference_type in self.inference_keys:
                for loss in losses:
                    total_loss[data_type][inference_type] += loss[data_type][
                        inference_type
                    ].item()

        # loss logging
        for data_type in self.data_keys:
            for inference_type in self.inference_keys:
                txt = f"{data_type}_{inference_type}_loss"
                self.logger.experiment.log(
                    {txt: total_loss[data_type][inference_type]}
                )

    def val_end_accuracy(
        self,
    ):
        # find max inference Accuracy
        for data_type in self.data_keys:
            for inference_type in self.inference_keys:
                accuracy = (
                    self.accuracy[data_type][inference_type].compute().item()
                )

                # Accuracy logging
                txt = f"{data_type}_{inference_type}_overall_accuracy"
                self.logger.experiment.log({txt: accuracy})

                if data_type == "valid":
                    if self.best_accuracy["overall"][inference_type] < accuracy:
                        self.best_accuracy["overall"][inference_type] = accuracy

                        if inference_type == "origin":
                            self.model.save_model(
                                self.default_root_dir, "Best_overall"
                            )
                    txt = f"[Best] valid_{inference_type}_overall_accuracy"
                    self.logger.experiment.log(
                        {txt: self.best_accuracy["overall"][inference_type]}
                    )

    def val_end_confusion_matrix(self):
        # ConfMat compute
        for data_type in self.data_keys:
            for inference_type in self.inference_keys:
                confusion_matrix = self.confusion_matrix[data_type][
                    inference_type
                ].compute()
                recall = self.specific_expression_accuracy_log(
                    data_type, inference_type, confusion_matrix
                )
                # average Accuracy logging
                txt = f"{data_type}_{inference_type}_average_accuracy"
                self.logger.experiment.log({txt: recall})

                if data_type == "valid":
                    if self.best_accuracy["average"][inference_type] < recall:
                        self.best_accuracy["average"][inference_type] = recall

                        self.log_confusion_matrix(
                            confusion_matrix, inference_type, measure="accuracy"
                        )
                        if inference_type == "origin":
                            self.model.save_model(
                                self.default_root_dir, "Best_average"
                            )

                    txt = f"[Best] valid_{inference_type}_average_accuracy"
                    self.logger.experiment.log(
                        {txt: self.best_accuracy["average"][inference_type]}
                    )

    def val_end_similarity(self):
        emotionNum = self.emotions
        batch_size = self.args["data_params"]["main"]["batch_size"]

        for data_type in self.data_keys:
            attention_matrix = self.attention_matrix[data_type]

            attention_matrix = (
                attention_matrix["value"] / attention_matrix["count"]
            )
            attention_matrix = attention_matrix.cpu().numpy()

            features = torch.cat(self.features[data_type])
            labels = torch.cat(self.labels[data_type])
            with torch.inference_mode():
                with torch.autocast(device_type=self.device.type):
                    cos_sim_mat = self.model.get_cos_sim_mat(features, 1.0)
            equal_mat = labels.view(-1, 1) == labels.view(1, -1)
            diff_mat = labels.view(-1, 1) != labels.view(1, -1)

            equal_similarity = cos_sim_mat[equal_mat].mean().item()
            txt = f"{data_type}_equal_similarity"
            self.logger.experiment.log({txt: equal_similarity})

            diff_similarity = cos_sim_mat[diff_mat].mean().item()
            txt = f"{data_type}_different_similarity"
            self.logger.experiment.log({txt: diff_similarity})

            if data_type == "valid":
                txt = "[Best] valid_equal_similarity"
                if self.best_similarity["equal"] < equal_similarity:
                    self.best_similarity["equal"] = equal_similarity
                    self.log_attention_matrix(
                        attention_matrix=attention_matrix
                        * batch_size
                        / emotionNum,
                        txt=txt,
                    )
                self.logger.experiment.log({txt: self.best_similarity["equal"]})

                txt = "[Best] valid_different_similarity"
                if self.best_similarity["different"] > diff_similarity:
                    self.best_similarity["different"] = diff_similarity
                    self.log_attention_matrix(
                        attention_matrix=attention_matrix
                        * batch_size
                        / emotionNum,
                        txt=txt,
                    )

                self.logger.experiment.log(
                    {txt: self.best_similarity["different"]}
                )

    def on_validation_end(self) -> None:
        # Loss
        self.val_end_losses(self.validation_step_outputs)

        # overall Accuracy
        self.val_end_accuracy()

        # average accuracy, confusion matrix, specific recall
        self.val_end_confusion_matrix()

        # attention & similarity
        self.val_end_similarity()

        ## Reset measure
        self.reset_accuracy_confusion_matrix()
        self.init_attention_matrix()

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
