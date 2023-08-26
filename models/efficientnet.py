"""
EfficientNet-b2 정의 모듈
"""

import os

import numpy as np
import torch
from torch import nn


class EfficientNet(nn.Module):
    """
    efficientnet-b2 for FER
    """

    def __init__(
        self,
        emotions: int,
        self_masking: bool,
        scale: float,
        sampling_ratio: float | None = None,
        n_positive: int | None = None,
        n_negative: int | None = None,
    ):
        """
        model initialize

        Args:
            emotions (int): n_classes for FER
            scale (float): scale factor (tau)
            self_masking (bool): using self-masking or not
            sampling_ratio (float): sampling ratio in ACL (gamma)
        """

        super().__init__()

        # model init
        self.model_name = "enet"

        if emotions == 7:
            self.backbone = torch.load("./weights/enet_b2_backbone_7emo.pth")
            self.classifier_att = torch.load(
                "./weights/enet_b2_classifier_7emo.pth"
            )
            self.classifier_main = torch.load(
                "./weights/enet_b2_classifier_7emo.pth"
            )
        elif emotions == 8:
            self.backbone = torch.load("./weights/enet_b2_backbone_8emo.pth")
            self.classifier_att = torch.load(
                "./weights/enet_b2_classifier_8emo.pth"
            )
            self.classifier_main = torch.load(
                "./weights/enet_b2_classifier_8emo.pth"
            )

        self.projection_head = nn.Sequential(
            nn.Linear(in_features=1408, out_features=1408),
            nn.ReLU(),
            nn.Linear(in_features=1408, out_features=1408),
        )

        # Attention param
        self.self_masking = self_masking
        self.scale = scale

        # Contrastive param
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.sampling_ratio = sampling_ratio

    def get_feature(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): input image

        Returns:
            torch.Tensor: CNN feature
        """

        return self.backbone(x)

    def get_cos_sim_mat(self, f: torch.Tensor, scale=None) -> torch.Tensor:
        """

        Args:
            f (torch.Tensor): feature

        Returns:
            torch.Tensor: scaled cosine similarity matrix
        """
        if not scale:
            scale = self.scale

        # feature to z
        z = self.projection_head(f)

        # dot product sim mat
        sim_mat = torch.matmul(z, z.T)

        norms = torch.norm(z, dim=1)
        norms = norms.view(1, -1)

        # cosine similarity matrix
        cos_sim_mat = sim_mat / torch.matmul(norms.T, norms)

        # scaling
        return cos_sim_mat / scale

    def forward_only_origin(self, x):
        feature = self.get_feature(x)
        output_tensor = self.classifier_main(feature)
        return output_tensor

    def forward(
        self, x: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """

        Args:
            x: input image tensor

        Returns:
            origin_output_tensor: model's traditional output
            attention_output_tensor: model's attention output
                                     made by referring other images in the batch
            att_mat: attention matrix
        """

        origin_feature = self.get_feature(x)
        origin_output_tensor = self.classifier_main(origin_feature)

        # origin feature로부터 cosine similarity matrix 생성
        cos_sim_mat = self.get_cos_sim_mat(origin_feature)

        # cosine similarity matrix 로부터 실제 feature 와 곱해질 attention matrix 생성
        att_mat = self.get_att_mat(cos_sim_mat, self.self_masking)

        # attention matrix와 origin feature 를 곱해 attention feature 생성
        attention_feature = torch.matmul(att_mat, origin_feature)

        # attention feature를 classifier_att 통과시켜 attention output 생성
        attention_output_tensor = self.classifier_att(attention_feature)

        return (
            origin_output_tensor,
            attention_output_tensor,
            att_mat,
            origin_feature,
        )

    def self_masking_matrix(
        self, matrix: torch.Tensor, replace: float = -10e6
    ) -> torch.Tensor:
        """

        Args:
            matrix (torch.Tensor): matrix whose diagonal will be -10e6

        Returns:
            torch.Tensor: self-masked similarity matrix
        """
        batch_size = matrix.shape[0]
        device = matrix.device
        data_type = matrix.dtype

        idx = np.diag_indices(batch_size)
        matrix[idx[0], idx[1]] = (
            (replace * torch.ones(batch_size, dtype=data_type))
            .to(device)
            .detach()
        )
        return matrix

    def get_att_mat(
        self, matrix: torch.Tensor, self_masking: bool
    ) -> torch.Tensor:
        """

        Args:
            matrix (torch.Tensor): scaled cosine similarity matrix
            self_masking (bool): is self masked?

        Returns:
            torch.Tensor: attention matrix applied softmax by row
        """

        # self masking
        if self_masking:
            matrix = self.self_masking_matrix(matrix)

        # get attention matrix
        attention_matrix = nn.Softmax(dim=1)(matrix)
        return attention_matrix

    def save_model(self, save_root: str, epoch: int | str) -> None:
        """
        save model's module

        Args:
            save_root (str): save_root
            epoch (int | str): epoch
        """

        backbone_path = os.path.join(
            save_root, f"{self.model_name}_backbone_{epoch:04}.pth"
        )
        projection_path = os.path.join(
            save_root, f"{self.model_name}_projection_{epoch:04}.pth"
        )
        classifier_main_path = os.path.join(
            save_root, f"{self.model_name}_classifier_main_{epoch:04}.pth"
        )
        classifier_att_path = os.path.join(
            save_root, f"{self.model_name}_classifier_att_{epoch:04}.pth"
        )

        torch.save(self.backbone, backbone_path)
        torch.save(self.projection_head, projection_path)
        torch.save(self.classifier_main, classifier_main_path)
        torch.save(self.classifier_att, classifier_att_path)

    def load_model(self, load_root: str, epoch: int | str) -> None:
        """
        load model's module

        Args:
            load_root (str): load_root
            epoch (int | str): epoch
        """

        backbone_path = os.path.join(
            load_root, f"{self.model_name}_backbone_{epoch:04}.pth"
        )
        projection_path = os.path.join(
            load_root, f"{self.model_name}_projection_{epoch:04}.pth"
        )
        classifier_main_path = os.path.join(
            load_root, f"{self.model_name}_classifier_main_{epoch:04}.pth"
        )
        classifier_att_path = os.path.join(
            load_root, f"{self.model_name}_classifier_att_{epoch:04}.pth"
        )

        self.backbone = torch.load(backbone_path)
        self.projection_head = torch.load(projection_path)
        self.classifier_main = torch.load(classifier_main_path)
        self.classifier_att = torch.load(classifier_att_path)

    """
    def get_positive(self, anchor, positives, negatives):
        images = torch.cat([anchor, positives, negatives])
        features = self.get_feature(images)
        features
        S = self.get_cossimMat(features)
        threshold = torch.quantile(S[0][1:], 1 - self.gamma)
        return S[0][1:] > threshold

    def get_negative(self, anchor, samples):
        with torch.no_grad():
            images = torch.cat([anchor, samples])
            f = self.get_feature(images)
            S = self.get_cossimMat(f)
            threshold = torch.quantile(S[0][1:], self.thres_per)
        return S[0][1:] < threshold

    def contrastive_loss(self, anchor, positives, negatives):
        images = torch.cat([anchor, positives, negatives])
        f = self.get_feature(images)
        S = self.get_cossimMat(f)
        # A = self.processing_A(S, self.att_mode)
        low = torch.exp(S[0][1:]).sum()
        up = torch.exp(S[0][1 : 1 + len(positives)]).sum()

        return -torch.log(up / low)

        # low = A[0][1:].sum()
        # up = A[0][1:1+len(positives)].sum()
        # low = nn.Softmax(dim=1)(S)[0][1:].sum()
        # up = nn.Softmax(dim=1)(S)[0][1:][A[0][1:]>0].sum()
    """
