from typing import Any
import torch
import os
from torch.utils.data import DataLoader, Dataset
import enum
from enum import auto
import pandas as pd
import numpy as np
import cv2
from PIL import Image

CHEXPERT_ROOT = '/disk1/wangjialei/datasets'
CHEXPERT_TRAIN_DATA_PATH = '/disk1/wangjialei/datasets/CheXpert-v1.0-small/train'
CHEXPERT_TRAIN_LABEL = '/disk1/wangjialei/datasets/CheXpert-v1.0-small/train.csv'
CHEXPERT_TEST_DATA_PATH = '/disk1/wangjialei/datasets/CheXpert-v1.0-small/valid'
CHEXPERT_TEST_LABEL = '/disk1/wangjialei/datasets/CheXpert-v1.0-small/valid.csv'


class CheXpert_Dataset(Dataset):
    def __init__(self, neg_one_to_zero: bool = False, transform=None):
        self.data = pd.read_csv(CHEXPERT_TRAIN_LABEL)[
                                ['Path', 'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].fillna(0)
        if neg_one_to_zero:
            self.data = self.data.replace(-1, 0)
        else:
            self.data = self.data.replace(-1, 1)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: Any) -> Any:
        item = Image.open(os.path.join(
            CHEXPERT_ROOT, self.data.iloc[index, 0]))
        if self.transform is not None:
            item = self.transform(item)
        return item, np.array(self.data.iloc[index, 1:], dtype=np.int8)


class CheXpert_TEST_Dataset(Dataset):
    def __init__(self, neg_one_to_zero: bool = False, transform=None):
        self.data = pd.read_csv(CHEXPERT_TEST_LABEL)[
                                ['Path', 'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].fillna(0)
        if neg_one_to_zero:
            self.data = self.data.replace(-1, 0)
        else:
            self.data = self.data.replace(-1, 1)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index: Any) -> Any:
        item=Image.open(os.path.join(CHEXPERT_ROOT,self.data.iloc[index,0]))
        if self.transform is not None:
            item=self.transform(item)
        return item,np.array(self.data.iloc[index,1:],dtype=np.int8),self.data.iloc[index,0]