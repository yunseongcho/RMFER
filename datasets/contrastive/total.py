import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import math

class celeba_CL(Dataset):
    def __init__(self, anno_root:str, data_root:str, neighborNum:int, skip_frame:int, transform=None) -> None:
        super().__init__()
        columns = ['img_path', 'identity']
        self.anno_df = pd.read_csv('/home/user/yscho/data/celeba/Anno/identity_CelebA.txt', sep=' ', names=columns)
        self.data_root = '/home/user/yscho/data/celeba/img_align_celeba'
        self.transform = transform
        self.skip_frame = skip_frame
        self.neighborNum = neighborNum

    def __len__(self):
        return len(self.anno_df)
    
    def __getitem__(self, idx):
        img_name, identity = self.anno_df.sample(1).values[0]
        anchor = Image.open(os.path.join(self.data_root, img_name))
        neg_df = self.anno_df[(self.anno_df.identity==identity)&(self.anno_df.img_path!=img_name)]
        if len(neg_df)<self.neighborNum:
            plus_df = self.anno_df[self.anno_df.identity!=identity].sample(self.neighborNum - len(neg_df))
            neg_df = pd.concat([neg_df, plus_df]).reset_index(drop=True)
        negative_imgPaths = list(neg_df.values[:, 0])
        negative_imgPaths = [os.path.join(self.data_root, str(i)) for i in negative_imgPaths]
        negatives = [Image.open(image_path) for image_path in negative_imgPaths]

        pos_df = self.anno_df[self.anno_df.identity!=identity]
        positive_imgPaths = list(pos_df.sample(self.neighborNum).values[:, 0])
        positive_imgPaths = [os.path.join(self.data_root, str(i)) for i in positive_imgPaths]
        positives = [Image.open(image_path) for image_path in positive_imgPaths]

        if self.transform:
            positives = torch.stack([self.transform(image) for image in positives])
            negatives = torch.stack([self.transform(image) for image in negatives])
            anchor = torch.stack([self.transform(anchor)])

        return anchor, positives, negatives
    
def get_celeba_CL_DataLoader(args):
    augments = transforms.RandomChoice([transforms.RandomGrayscale(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(
                                            brightness=0.25,
                                            contrast=0.25,
                                            saturation=0.25,
                                            hue=0.125
                                            )])
    
    RMset_normalize = transforms.Normalize(mean=[0.61120795, 0.47080194, 0.40948687],
                                            std=[0.21088221, 0.1876408,  0.17565419])
    
    if args['data_params']['main']['dataset']=='FERPlus':
        train_transforms = transforms.Compose([
            transforms.Resize((260, 260)), 
            transforms.Grayscale(num_output_channels=3),
            augments,
            transforms.ToTensor(), 
            RMset_normalize])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((260, 260)), 
            augments,
            transforms.ToTensor(), 
            RMset_normalize])
    
    dataset = celeba_CL(anno_root=args['data_params']['RMset']['anno_root'],
                    data_root=args['data_params']['RMset']['data_root'],
                    neighborNum=args['data_params']['RMset']['neighborNum'],
                    skip_frame=args['data_params']['RMset']['skip_frame'],
                    transform=train_transforms)

    dataloader = DataLoader(dataset, batch_size=args['data_params']['RMset']['batch_size'],
                            num_workers=args['data_params']['RMset']['num_workers'], drop_last=True)

    return dataloader

class AffNet_CL(Dataset):
    def __init__(self, anno_root:str, data_root:str, neighborNum:int, skip_frame:int, transform=None) -> None:
        super().__init__()
        self.anno_df = pd.read_csv('/home/user/yscho/data/AffectNet_preprocessed_by_RetinaFace/training.csv')
        self.data_root = '/home/user/yscho/data/AffectNet_preprocessed_by_RetinaFace/images'
        self.transform = transform
        self.skip_frame = skip_frame
        self.neighborNum = neighborNum

    def __len__(self):
        return len(self.anno_df)
    
    def __getitem__(self, idx):
        anchor = Image.open(os.path.join(self.data_root, self.anno_df.sample(1).values[0][0]))
        negative_imgPaths = list(self.anno_df.sample(self.neighborNum).values[:, 0])
        negative_imgPaths = [os.path.join(self.data_root, str(i)) for i in negative_imgPaths]
        negatives = [Image.open(image_path) for image_path in negative_imgPaths]

        positive_imgPaths = list(self.anno_df.sample(self.neighborNum).values[:, 0])
        positive_imgPaths = [os.path.join(self.data_root, str(i)) for i in positive_imgPaths]
        positives = [Image.open(image_path) for image_path in positive_imgPaths]

        if self.transform:
            positives = torch.stack([self.transform(image) for image in positives])
            negatives = torch.stack([self.transform(image) for image in negatives])
            anchor = torch.stack([self.transform(anchor)])

        return anchor, positives, negatives
    
def get_AffNet_CL_DataLoader(args):
    augments = transforms.RandomChoice([transforms.RandomGrayscale(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(
                                            brightness=0.25,
                                            contrast=0.25,
                                            saturation=0.25,
                                            hue=0.125
                                            )])
    
    RMset_normalize = transforms.Normalize(mean=[0.61120795, 0.47080194, 0.40948687],
                                            std=[0.21088221, 0.1876408,  0.17565419])
    
    if args['data_params']['main']['dataset']=='FERPlus':
        train_transforms = transforms.Compose([
            transforms.Resize((260, 260)), 
            transforms.Grayscale(num_output_channels=3),
            augments,
            transforms.ToTensor(), 
            RMset_normalize])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((260, 260)), 
            augments,
            transforms.ToTensor(), 
            RMset_normalize])
    
    dataset = AffNet_CL(anno_root=args['data_params']['RMset']['anno_root'],
                    data_root=args['data_params']['RMset']['data_root'],
                    neighborNum=args['data_params']['RMset']['neighborNum'],
                    skip_frame=args['data_params']['RMset']['skip_frame'],
                    transform=train_transforms)

    dataloader = DataLoader(dataset, batch_size=args['data_params']['RMset']['batch_size'],
                            num_workers=args['data_params']['RMset']['num_workers'], drop_last=True)

    return dataloader


class RMset_noassumption(Dataset):
    def __init__(self, anno_root:str, data_root:str, neighborNum:int, skip_frame:int, transform=None) -> None:
        super().__init__()
        self.anno_df = pd.read_pickle('/home/yscho/D/RMset/annotations/annotation_noAssumption.pkl')
        self.data_root = '/SSD/RMset/image/Ada-CM/'
        self.transform = transform
        self.skip_frame = skip_frame
        self.neighborNum = neighborNum

    def __len__(self):
        return 2160000
    
    def __getitem__(self, idx):
        anchor = Image.open(os.path.join(self.data_root, self.anno_df.sample(1).values[0][0]))
        negative_imgPaths = list(self.anno_df.sample(self.neighborNum).values.flatten())
        negative_imgPaths = [os.path.join(self.data_root, str(i)) for i in negative_imgPaths]
        negatives = [Image.open(image_path) for image_path in negative_imgPaths]

        positive_imgPaths = list(self.anno_df.sample(self.neighborNum).values.flatten())
        positive_imgPaths = [os.path.join(self.data_root, str(i)) for i in positive_imgPaths]
        positives = [Image.open(image_path) for image_path in positive_imgPaths]

        if self.transform:
            positives = torch.stack([self.transform(image) for image in positives])
            negatives = torch.stack([self.transform(image) for image in negatives])
            anchor = torch.stack([self.transform(anchor)])

        return anchor, positives, negatives
    
def get_RMset_noAssume_DataLoader(args):
    augments = transforms.RandomChoice([transforms.RandomGrayscale(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(
                                            brightness=0.25,
                                            contrast=0.25,
                                            saturation=0.25,
                                            hue=0.125
                                            )])
    
    RMset_normalize = transforms.Normalize(mean=[0.52934384, 0.39199043, 0.3694192 ],
                                            std=[0.20445083, 0.17351717, 0.16180859])
    
    if args['data_params']['main']['dataset']=='FERPlus':
        train_transforms = transforms.Compose([
            transforms.Resize((260, 260)), 
            transforms.Grayscale(num_output_channels=3),
            augments,
            transforms.ToTensor(), 
            RMset_normalize])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((260, 260)), 
            augments,
            transforms.ToTensor(), 
            RMset_normalize])
    
    dataset = RMset_noassumption(anno_root=args['data_params']['RMset']['anno_root'],
                    data_root=args['data_params']['RMset']['data_root'],
                    neighborNum=args['data_params']['RMset']['neighborNum'],
                    skip_frame=args['data_params']['RMset']['skip_frame'],
                    transform=train_transforms)

    dataloader = DataLoader(dataset, batch_size=args['data_params']['RMset']['batch_size'],
                            num_workers=args['data_params']['RMset']['num_workers'], drop_last=True)

    return dataloader
    


class RMset(Dataset):

    def __init__(self, anno_root:str, data_root:str, neighborNum:int, skip_frame:int, transform=None) -> None:
        super().__init__()
        self.anno_root = anno_root
        self.data_root = data_root
        self.transform = transform
        self.skip_frame = skip_frame
        self.neighborNum = neighborNum

    def __len__(self):
        return 40000000

    def __getitem__(self, idx):
        annotations = os.listdir(self.anno_root)
        annotation = random.sample(annotations, 1)[0]

        anno_df = pd.read_pickle(os.path.join(self.anno_root, annotation))
        
        remain = random.randrange(self.skip_frame)
        anno_df = anno_df[anno_df.frameNum%self.skip_frame==remain].reset_index(drop=True)
        anno_df.frameNum = anno_df.frameNum.to_numpy()//self.skip_frame
        
        faceNum = len(anno_df.faceID.drop_duplicates())
        faceID = random.randrange(faceNum)
        
        frameRange = math.ceil((self.neighborNum / faceNum-1)/2) + 1
        frameNums = list(anno_df.frameNum.drop_duplicates().sort_values())
    
        while True:
            frameNum = random.randint(31, len(frameNums)-31)
            anchor_df = anno_df[(anno_df.faceID==faceID)&(anno_df.frameNum==frameNum)]
            positive_df = anno_df[(anno_df.frameNum>=frameNum-frameRange)&(anno_df.frameNum<frameNum+frameRange)&(anno_df.faceID!=faceID)]
            if (len(positive_df)>=self.neighborNum) & (not anchor_df.empty):
                break
        
        negative_df = anno_df[(anno_df.faceID==faceID)]
        negative_df = negative_df[(negative_df.frameNum<=frameNum-30)|(negative_df.frameNum>frameNum+30)]
        negative_imgPaths = [random.choice(negative_df.filePath.tolist()) for i in range(self.neighborNum)]# random.sample(, k=)
        negative_imgPaths = [os.path.join(self.data_root, str(i)) for i in negative_imgPaths]
        negatives = [Image.open(image_path) for image_path in negative_imgPaths]
        
        positive_imgPaths = random.sample(positive_df.filePath.tolist(), k=self.neighborNum)
        positive_imgPaths = [os.path.join(self.data_root, str(i)) for i in positive_imgPaths]
        positives = [Image.open(image_path) for image_path in positive_imgPaths]

        anchor = Image.open(os.path.join(self.data_root, anchor_df.filePath.values[0]))

        if self.transform:
            positives = torch.stack([self.transform(image) for image in positives])
            negatives = torch.stack([self.transform(image) for image in negatives])
            anchor = torch.stack([self.transform(anchor)])

        return anchor, positives, negatives

def get_RMset_DataLoader(args):
    augments = transforms.RandomChoice([transforms.RandomGrayscale(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(
                                            brightness=0.25,
                                            contrast=0.25,
                                            saturation=0.25,
                                            hue=0.125
                                            )])
    
    RMset_normalize = transforms.Normalize(mean=[0.52934384, 0.39199043, 0.3694192 ],
                                            std=[0.20445083, 0.17351717, 0.16180859])
    
    if args['data_params']['main']['dataset']=='FERPlus':
        train_transforms = transforms.Compose([
            transforms.Resize((260, 260)), 
            transforms.Grayscale(num_output_channels=3),
            augments,
            transforms.ToTensor(), 
            RMset_normalize])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((260, 260)), 
            augments,
            transforms.ToTensor(), 
            RMset_normalize])
    
    dataset = RMset(anno_root=args['data_params']['RMset']['anno_root'],
                    data_root=args['data_params']['RMset']['data_root'],
                    neighborNum=args['data_params']['RMset']['neighborNum'],
                    skip_frame=args['data_params']['RMset']['skip_frame'],
                    transform=train_transforms)

    dataloader = DataLoader(dataset, batch_size=args['data_params']['RMset']['batch_size'],
                            num_workers=args['data_params']['RMset']['num_workers'], drop_last=True)

    return dataloader