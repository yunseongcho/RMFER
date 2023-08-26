# pylint: disable=invalid-name
"""
define of Reaction Mashup dataset of contrastive learning
RMset class
"""

import math
import os
import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class RMset(Dataset):
    """
    define RMset class

    RMset contains many annotation file per a video.

    """

    def __init__(
        self,
        anno_root: str,
        data_root: str,
        n_positive: int,
        n_negative: int,
        n_frame_skip: int,
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__()

        self.anno_root = f"{anno_root}/{n_frame_skip:02}"
        self.anno_names = os.listdir(self.anno_root)
        self.data_root = data_root
        self.transform = transform

        # for contrastive
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.n_frame_skip = n_frame_skip

    def __len__(self):
        # very big number
        return int(10e10)

    def __getitem__(self, idx):
        # random select annotation file
        anno_name = random.sample(self.anno_names, 1)[0]
        anno_df = pd.read_pickle(os.path.join(self.anno_root, anno_name))

        faceNum = len(anno_df.faceID.drop_duplicates())
        faceID = random.randrange(faceNum)

        frameRange = math.ceil((self.neighborNum / faceNum - 1) / 2) + 1
        frameNums = list(anno_df.frameNum.drop_duplicates().sort_values())

        while True:
            frameNum = random.randint(31, len(frameNums) - 31)
            anchor_df = anno_df[
                (anno_df.faceID == faceID) & (anno_df.frameNum == frameNum)
            ]
            positive_df = anno_df[
                (anno_df.frameNum >= frameNum - frameRange)
                & (anno_df.frameNum < frameNum + frameRange)
                & (anno_df.faceID != faceID)
            ]
            if (len(positive_df) >= self.neighborNum) & (not anchor_df.empty):
                break

        negative_df = anno_df[(anno_df.faceID == faceID)]
        negative_df = negative_df[
            (negative_df.frameNum <= frameNum - 30)
            | (negative_df.frameNum > frameNum + 30)
        ]
        negative_imgPaths = [
            random.choice(negative_df.filePath.tolist())
            for i in range(self.neighborNum)
        ]  # random.sample(, k=)
        negative_imgPaths = [
            os.path.join(self.data_root, str(i)) for i in negative_imgPaths
        ]
        negatives = [Image.open(image_path) for image_path in negative_imgPaths]

        positive_imgPaths = random.sample(
            positive_df.filePath.tolist(), k=self.neighborNum
        )
        positive_imgPaths = [
            os.path.join(self.data_root, str(i)) for i in positive_imgPaths
        ]
        positives = [Image.open(image_path) for image_path in positive_imgPaths]

        anchor = Image.open(
            os.path.join(self.data_root, anchor_df.filePath.values[0])
        )

        if self.transform:
            positives = torch.stack(
                [self.transform(image) for image in positives]
            )
            negatives = torch.stack(
                [self.transform(image) for image in negatives]
            )
            anchor = torch.stack([self.transform(anchor)])

        return anchor, positives, negatives


def get_RMset_DataLoader(args):
    augments = transforms.RandomChoice(
        [
            transforms.RandomGrayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.125
            ),
        ]
    )

    RMset_normalize = transforms.Normalize(
        mean=[0.52934384, 0.39199043, 0.3694192],
        std=[0.20445083, 0.17351717, 0.16180859],
    )

    if args["data_params"]["main"]["dataset"] == "FERPlus":
        train_transforms = transforms.Compose(
            [
                transforms.Resize((260, 260)),
                transforms.Grayscale(num_output_channels=3),
                augments,
                transforms.ToTensor(),
                RMset_normalize,
            ]
        )
    else:
        train_transforms = transforms.Compose(
            [
                transforms.Resize((260, 260)),
                augments,
                transforms.ToTensor(),
                RMset_normalize,
            ]
        )

    dataset = RMset(
        anno_root=args["data_params"]["RMset"]["anno_root"],
        data_root=args["data_params"]["RMset"]["data_root"],
        neighborNum=args["data_params"]["RMset"]["neighborNum"],
        skip_frame=args["data_params"]["RMset"]["skip_frame"],
        transform=train_transforms,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args["data_params"]["RMset"]["batch_size"],
        num_workers=args["data_params"]["RMset"]["num_workers"],
        drop_last=True,
    )

    return dataloader
