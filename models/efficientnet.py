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

    def __init__(self, emotions: int = 7, self_mask: bool = True, gamma: float = 0.1, scale: float = 0.25):
        super().__init__()

        # model init
        if emotions == 7:
            self.backbone = torch.load("./weights/enet_b2_backbone_7emo.pth")
            self.classifier_att = torch.load("./weights/enet_b2_classifier_7emo.pth")
            self.classifier_main = torch.load("./weights/enet_b2_classifier_7emo.pth")
        elif emotions == 8:
            self.backbone = torch.load("./weights/enet_b2_backbone_8emo.pth")
            self.classifier_att = torch.load("./weights/enet_b2_classifier_8emo.pth")
            self.classifier_main = torch.load("./weights/enet_b2_classifier_8emo.pth")

        self.projection_head = nn.Sequential(
            nn.Linear(in_features=1408, out_features=1408), nn.ReLU(), nn.Linear(in_features=1408, out_features=1408)
        )

        # Attention param
        self.self_mask = self_mask
        self.scale = scale

        # Contrastive param
        self.gamma = gamma

    def get_feature(self, x):
        return self.backbone(x)

    def get_cos_sim_mat(self, f):
        z = self.projection_head(f)
        similarity_matrix = torch.matmul(z, z.T)
        norms = torch.norm(z, dim=1)
        norms = norms.view(1, -1)
        return (similarity_matrix / torch.matmul(norms.T, norms)) / self.scale

    def forward(self, x):
        origin_feature = self.get_feature(x)
        origin_output_tensor = self.classifier_main(origin_feature)

        cos_sim_matrix = self.get_cos_sim_mat(origin_feature)
        attention_matrix = self.get_att_mat(cos_sim_matrix, self.self_mask)

        attention_feature = torch.matmul(attention_matrix, origin_feature)
        attention_output_tensor = self.classifier_att(attention_feature)

        return origin_output_tensor, attention_output_tensor, attention_matrix

    def self_masking(self, matrix):
        idx = np.diag_indices(matrix.shape[0])
        matrix[idx[0], idx[1]] = (-10e6 * torch.ones(matrix.shape[0])).to(matrix.device).detach()
        return matrix

    def get_att_mat(self, matrix, self_masking):
        if self_masking:
            matrix = self.self_masking(matrix)

        attention_matrix = nn.Softmax(dim=1)(matrix)
        return attention_matrix

    def get_positive(self, anchor, samples):
        with torch.no_grad():
            images = torch.cat([anchor, samples])
            f = self.get_feature(images)
            S = self.get_cossimMat(f)
            threshold = torch.quantile(S[0][1:], 1 - self.thres_per)
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

    def save_model(self, PATH, epoch):
        backbone_path = os.path.join(PATH, "backbone_{}.pth".format(epoch))
        projection_path = os.path.join(PATH, "projection_{}.pth".format(epoch))
        classifier_main_path = os.path.join(PATH, "classifier_main_{}.pth".format(epoch))
        classifier_att_path = os.path.join(PATH, "classifier_att_{}.pth".format(epoch))

        torch.save(self.backbone.state_dict(), backbone_path)
        torch.save(self.projection_head.state_dict(), projection_path)
        torch.save(self.classifier_main.state_dict(), classifier_main_path)
        torch.save(self.classifier_att.state_dict(), classifier_att_path)

    def load_model(self, PATH, epoch):
        backbone_path = os.path.join(PATH, "backbone_{}.pth".format(epoch))
        projection_path = os.path.join(PATH, "projection_{}.pth".format(epoch))
        classifier_main_path = os.path.join(PATH, "classifier_main_{}.pth".format(epoch))
        classifier_att_path = os.path.join(PATH, "classifier_att_{}.pth".format(epoch))

        self.backbone.load_state_dict(torch.load(backbone_path))
        self.projection_head.load_state_dict(torch.load(projection_path))
        self.classifier_main.load_state_dict(torch.load(classifier_main_path))
        self.classifier_att.load_state_dict(torch.load(classifier_att_path))
