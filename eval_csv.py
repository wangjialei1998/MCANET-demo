import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

import numpy as np
from timm.data import create_loader
from timm.utils import *
from timm.models import create_model
# from timm.data import LoadImagesAndLabels,preprocess,LoadImagesAndLabelsV2,LoadImagesAndSoftLabels
from Pytorch_RIADD_main.timm.data import get_riadd_train_transforms,get_riadd_valid_transforms,get_riadd_test_transforms,get_valid_transforms
from ODIR_DATASET import ODIR_TEST_DATASET,TEST_DATA_PATH,TRAIN_DATA_PATH,ODIR_VALID_DATASET,VALID_DATA_PATH,TEST_DATA_PATH
import os
from tqdm import tqdm
import random
import torch.distributed as dist
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

CFG = {
    'seed': 42,
    'img_size': 384,
    'valid_bs': 128,
    'num_workers': 16,
    'num_classes': 8,
    'tta': 4,
    'weights': [1,1,1,1]
}



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def validate(model, loader): 
    model.eval()
    preds = []
    names=[]
    pbar = tqdm(enumerate(loader), total=len(loader))  
    with torch.no_grad():
        for batch_idx, (input, target,name) in pbar:
            input = input.cuda()
            output = model(input)
            preds.append(output.sigmoid().to('cpu').numpy())
            names=names+list(name)
    predictions = np.concatenate(preds)
    return predictions,names

if __name__ == '__main__':
    from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold
    import pandas as pd
    import torch.utils.data as data
    seed_everything(CFG['seed'])
    test_index = ['ID','N','D','G','C','A','H','M','O']
    test_transforms = get_valid_transforms(image_size=CFG['img_size'])
    test_dataset = ODIR_TEST_DATASET(transform=test_transforms)
    # test_data_loader = data.DataLoader( test_dataset, 
    #                                     batch_size=CFG['valid_bs'], 
    #                                     shuffle=False, 
    #                                     num_workers=CFG['num_workers'], 
    #                                     pin_memory=True, 
    #                                     drop_last=False,
    #                                     sampler = None)


    loader_eval = create_loader(
        test_dataset,
        input_size=(3,CFG['img_size'],CFG['img_size']),
        batch_size=CFG['valid_bs'],
        is_training=False,
        use_prefetcher=False,
        interpolation='bicubic',
        mean=(0.4407552, 0.28228086, 0.15446076),
        std=(0.254417, 0.17148255, 0.0995115),
        num_workers=CFG['num_workers'],
        distributed=False,
        crop_pct=False,
        pin_memory=0
    )
    test = pd.DataFrame(columns=test_index)

    tst_preds = []
    model_path = ['/disk1/wangjialei/research/odir_main/all_fusionloss/train/20230702-100809-res2net50d-384/checkpoint-23.pth.tar',
                  '/disk1/wangjialei/research/odir_main/all_bce_before/train/20230702-114441-davit_base-384/checkpoint-25.pth.tar',
                  '/disk1/wangjialei/research/odir_main/all_fusionloss/train/20230702-104709-resnet50d-384/checkpoint-24.pth.tar',
                  '/disk1/wangjialei/research/odir_main/all_fusionloss/train/20230702-100809-res2net50d-384/checkpoint-24.pth.tar']
    model_names=['res2net50d','davit_base','resnet50d','res2net50d']
    for i in range(CFG['tta']):
        model = create_model(model_name = model_names[i],num_classes=CFG['num_classes'])
        print('model_path: ',model_path[i])
        state_dict = torch.load(model_path[i],map_location='cpu')
        model.load_state_dict(state_dict["state_dict"], strict=True)
        # model = nn.DataParallel(model)
        model.cuda()
        results,images=validate(model,loader_eval)
        tst_preds += [1/CFG['tta']*results]
    tst_preds = np.sum(tst_preds, axis=0)
    test['ID']=images
    test[test_index[1:]] = tst_preds
    test.to_csv('./test.csv', index=False)
