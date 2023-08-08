from typing import Any
import torch
import os
from torch.utils.data import DataLoader, Dataset
import enum
from enum import auto
import pandas as pd
import numpy as np
# import cv2 as cv
from PIL import Image

ODIR_ROOT = '/disk1/wangjialei/datasets/odir_new'
TRAIN_DATA_PATH='/disk1/wangjialei/datasets/OIA-ODIR/Training_Set/Images_CROP'
TRAIN_LABEL='/disk1/wangjialei/datasets/OIA-ODIR/Training_Set/Annotation/training_annotation_(English)_single.csv'
VALID_DATA_PATH='/disk1/wangjialei/datasets/OIA-ODIR/Off_site_Test_Set/Images_CROP'
VALID_LABEL='/disk1/wangjialei/datasets/OIA-ODIR/Off_site_Test_Set/Annotation/off_site_test_annotation_(English)_single_for_submit.csv'
TEST_DATA_PATH='/disk1/wangjialei/datasets/OIA-ODIR/On_site_Test_Set/Images_CROP'
TEST_LABEL='/disk1/wangjialei/datasets/OIA-ODIR/On_site_Test_Set/Annotation/on_site_test_annotation_(English)_single_for_submit.csv'
class ODIR_DATASET(Dataset):
    def __init__(self,images_names:pd.DataFrame, transform=None) -> None:
        super().__init__()
        self.images_names=images_names
        self.data_path = TRAIN_DATA_PATH
        self.images = os.listdir(self.data_path)
        self.transform = transform

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, index: Any) -> Any:
        item = Image.open(os.path.join(self.data_path, self.images_names.iloc[index,0]))
        if self.transform is not None:
            item = self.transform(item)
        return item, np.array(self.images_names.iloc[index,1:], dtype=np.int8)


class ODIR_TEST_DATASET(Dataset):
    def __init__(self,transform=None) -> None:
        super().__init__()
        self.data_names=pd.read_csv(TEST_LABEL)
        self.transform=transform
    def __len__(self):
        return len(self.data_names)
    def __getitem__(self, index: Any) -> Any:
        item=Image.open(os.path.join(TEST_DATA_PATH,self.data_names.iloc[index,0]))
        if self.transform is not None:
            item=self.transform(item)
        return item,np.array(self.data_names.iloc[index,1:], dtype=np.int8),self.data_names.iloc[index,0]
class ODIR_VALID_DATASET(Dataset):
    def __init__(self,transform=None) -> None:
        super().__init__()
        self.data_names=pd.read_csv(VALID_LABEL)
        self.transform=transform
    def __len__(self):
        return len(self.data_names)
    def __getitem__(self, index: Any) -> Any:
        item=Image.open(os.path.join(VALID_DATA_PATH,self.data_names.iloc[index,0]))
        if self.transform is not None:
            item=self.transform(item)
        return item, np.array(self.data_names.iloc[index,1:], dtype=np.int8),self.data_names.iloc[index,0]
    

class ODIR_DATASET_TRAIN_FOR_STACKING(Dataset):
    def __init__(self, transform=None) -> None:
        super().__init__()
        self.images_names=pd.read_csv(TRAIN_LABEL)
        self.data_path = TRAIN_DATA_PATH
        self.images = os.listdir(self.data_path)
        self.transform = transform

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, index: Any) -> Any:
        item = Image.open(os.path.join(self.data_path, self.images_names.iloc[index,0]))
        if self.transform is not None:
            item = self.transform(item)
        return item, np.array(self.images_names.iloc[index,1:], dtype=np.int8),self.images_names.iloc[index,0]