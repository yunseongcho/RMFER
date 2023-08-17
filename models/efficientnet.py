"""
EfficientNet-b2 정의 모듈
"""

import os

import numpy as np
import timm
import torch
from torch import nn


class EfficientNet(nn.Module):
    """
    efficientnet-b2 for FER
    """

    def __init__(self, emotions: int = 7, self_mask: bool = True, gamma: float = 0.1, scale: float = 0.25):
        super().__init__()

        # Base model 정의
        model = timm.create_model("tf_efficientnet_b2_ns", pretrained=False)
        if emotions == 7:
            model.classifier = nn.Linear(in_features=1408, out_features=7)
            model.load_state_dict(torch.load("./enet_b2_7.pt").state_dict())
        elif emotions == 8:
            model.classifier = nn.Linear(in_features=1408, out_features=8)
            model.load_state_dict(torch.load("./enet_b2_8.pt").state_dict())

        modules = list(model.children())[:-1]

        # backbone 정의
        self.backbone = nn.Sequential(*modules)

        self.projection_head = nn.Sequential(
            nn.Linear(in_features=1408, out_features=1408), nn.ReLU(), nn.Linear(in_features=1408, out_features=1408)
        )
        self.classifier_att = nn.Linear(in_features=1408, out_features=emotions)
        self.classifier_main = list(model.children())[-1]

        # Attention param
        self.self_mask = self_mask
        self.scale = scale

        # Contrastive param
        self.gamma = gamma

    def forward(self, x):
        origin_f = self.get_feature(x)
        origin_outputs = self.classifier_main(origin_f)

        S = self.get_cossimMat(origin_f)
        A = self.processing_A(S, self.att_mode)

        att_f = torch.matmul(A, origin_f)
        # use or not classifier_att
        if self.useClsAtt:
            att_outputs = self.classifier_att(att_f)
        else:
            att_outputs = self.classifier_main(att_f)
        return origin_outputs, att_outputs, A

    def get_feature(self, x):
        return self.backbone(x)

    def get_cossimMat(self, f):
        # use or not projection head
        if self.useHead:
            z = self.projection_head(f)
        else:
            z = f
        S = torch.matmul(z, z.T)
        norms = torch.norm(z, dim=1)
        norms = norms.view(1, -1)
        return (S / torch.matmul(norms.T, norms)) / self.scale

    def softmax_A(self, A):
        A = nn.Softmax(dim=1)(A)
        return A

    def delself_A(self, A):
        idx = np.diag_indices(A.shape[0])
        A[idx[0], idx[1]] = (-10e6 * torch.ones(A.shape[0])).to(A.device).detach()
        return A

    def processing_A(self, A, mode):
        if mode == "s":
            A = self.softmax_A(A)
        elif mode == "ds":
            A = self.softmax_A(self.delself_A(A))
        return A

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
