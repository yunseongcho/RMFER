# pylint: disable=invalid-name
"""
define AffectNet benchmark dataset
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


class AffectNet(Dataset):
    """
    AffectNet dataset class
    """

    def __init__(
        self,
        anno_path: str,
        data_root: str,
        emotions: int,
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__()

        # annotation DataFrame 읽기
        self.annotation = pd.read_csv(anno_path)

        # emotion 개수에 포함되는 sample들만 남기기
        self.emotions = emotions
        self.annotation = self.annotation[
            self.annotation.expression < emotions
        ].reset_index(drop=True)

        # WeightedRandomSampler에 사용할 target들
        self.targets = self.annotation.expression.to_numpy()

        # 이미지들의 root
        self.data_root = data_root

        # transform
        self.transform = transform

    def __len__(self):
        # 데이터셋의 전체 길이는 annotation의 길이
        return len(self.annotation)

    def __getitem__(self, idx):
        # index가 tensor라면 list로 변환시킨다
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 해당 index의 img들의 경로를 가져오고 img를 open한다.
        image_path = os.path.join(
            self.data_root, self.annotation.subDirectory_filePath[idx]
        )
        image = Image.open(image_path)

        # 설정된 transform이 있을 경우 변환을 시켜준다.
        if self.transform:
            image = self.transform(image)

        # label이 될 expression을 받아온다.
        expression = np.array(self.annotation.expression[idx])
        return image, expression


def get_weighted_random_sampler(dataset):
    # sampling 할 weight 생성
    target = dataset.targets
    class_sample_count = np.unique(target, return_counts=True)[
        1
    ]  # target 별 cnt 수
    weight = 1.0 / class_sample_count  # target 별 weight

    # 만약 데이터셋 내에 빠진 label이 있다면
    Labels = set(range(dataset.emotions))
    nonsexists = list(Labels - set(target))  # 데이터셋 내 빠진 label
    if nonsexists:
        nonsexists.sort()
        for idx in nonsexists:
            weight = np.insert(
                weight, idx, 0
            )  # weight에 순차적으로 label idx에 weight(0) 추가

    # sample 별 weight 추가
    samples_weight = weight[target]
    samples_weight = torch.from_numpy(samples_weight)

    # WeightedRandomSampler return
    return WeightedRandomSampler(samples_weight, len(samples_weight))


def get_AffectNet_DataLoader(args, isTrain: bool):
    augments = transforms.RandomChoice(
        [
            transforms.RandomGrayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.125
            ),
        ]
    )

    train_normalize = transforms.Normalize(
        mean=[0.61120795, 0.47080194, 0.40948687],
        std=[0.21088221, 0.1876408, 0.17565419],
    )

    valid_normalize = transforms.Normalize(
        mean=[0.59882072, 0.46099265, 0.40169688],
        std=[0.20795445, 0.18521879, 0.17365596],
    )

    train_transforms = transforms.Compose(
        [
            transforms.Resize((260, 260)),
            augments,
            transforms.ToTensor(),
            train_normalize,
        ]
    )

    valid_transforms = transforms.Compose(
        [transforms.Resize((260, 260)), transforms.ToTensor(), valid_normalize]
    )

    # return train DataLoader
    if isTrain:
        dataset = AffectNet(
            anno_path=args["data_params"]["main"]["train_anno_path"],
            data_root=args["data_params"]["main"]["data_root"],
            emotions=args["exp_params"]["emotions"],
            transform=train_transforms,
        )
        sampler = get_weighted_random_sampler(dataset)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args["data_params"]["main"]["batch_size"],
            num_workers=args["data_params"]["main"]["num_workers"],
            sampler=sampler,
            drop_last=True,
        )

    # return valid DataLoader
    else:
        dataset = AffectNet(
            anno_path=args["data_params"]["main"]["val_anno_path"],
            data_root=args["data_params"]["main"]["data_root"],
            emotions=args["exp_params"]["emotions"],
            transform=valid_transforms,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args["data_params"]["main"]["batch_size"],
            num_workers=args["data_params"]["main"]["num_workers"],
            shuffle=False,
            drop_last=False,
        )

    return dataloader
