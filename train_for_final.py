import torch
import os
import timm
import time

os.environ['CUDA_VISIBLE_DEVICES']='5'
model=timm.create_model('davit_base')
model.cuda()
a=torch.ones((20,3,384,384)).cuda()
while True:
    model(a)
    time.sleep(2)