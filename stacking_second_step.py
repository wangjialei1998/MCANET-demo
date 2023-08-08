from typing import Any
import torch
from torch.utils.data import DataLoader,Dataset
from torch import nn
import pandas as pd
import os
import numpy as np
from timm.scheduler import create_scheduler
from timm.utils import AverageMeter
from sklearn.metrics import average_precision_score, recall_score, f1_score, precision_score
from torch.utils.tensorboard import SummaryWriter
from evaluation import ODIR_Metrics
import timm
from timm import create_model
class Label_Dataset4(Dataset):
    def __init__(self,flatten:bool,is_test:bool) -> None:
        super().__init__()
        self.data=create_data(is_test=is_test)
        self.flatten=flatten
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: Any) -> Any:
        result=np.array(self.data.iloc[index,1:33],dtype=np.float32)
        target=np.array(self.data.iloc[index,33:],dtype=np.float32)
        if not self.flatten:
            result=result.reshape(4,8)
        return np.expand_dims(result,axis=0),target

def create_data(is_test:bool):
    if is_test:
        data1=pd.read_csv('/disk1/wangjialei/research/odir_main/stacking/train1.csv')[['ID','N','D','G','C','A','H','M','O']]
        data2=pd.read_csv('/disk1/wangjialei/research/odir_main/stacking/train2.csv')[['ID','N','D','G','C','A','H','M','O']]
        data3=pd.read_csv('/disk1/wangjialei/research/odir_main/stacking/train3.csv')[['ID','N','D','G','C','A','H','M','O']]
        data4=pd.read_csv('/disk1/wangjialei/research/odir_main/stacking/train4.csv')
        newdata=pd.merge(data1,data2,on='ID')
        newdata=pd.merge(newdata,data3,on='ID')
        newdata=pd.merge(newdata,data4,on='ID')

        data1=pd.read_csv('/disk1/wangjialei/research/odir_main/stacking/valid1.csv')[['ID','N','D','G','C','A','H','M','O']]
        data2=pd.read_csv('/disk1/wangjialei/research/odir_main/stacking/valid2.csv')[['ID','N','D','G','C','A','H','M','O']]
        data3=pd.read_csv('/disk1/wangjialei/research/odir_main/stacking/valid3.csv')[['ID','N','D','G','C','A','H','M','O']]
        data4=pd.read_csv('/disk1/wangjialei/research/odir_main/stacking/valid4.csv')
        newdata1=pd.merge(data1,data2,on='ID')
        newdata1=pd.merge(newdata1,data3,on='ID')
        newdata1=pd.merge(newdata1,data4,on='ID')
        return newdata.append(newdata1)
    else:
        data1=pd.read_csv('/disk1/wangjialei/research/odir_main/stacking/test1.csv')[['ID','N','D','G','C','A','H','M','O']]
        data2=pd.read_csv('/disk1/wangjialei/research/odir_main/stacking/test2.csv')[['ID','N','D','G','C','A','H','M','O']]
        data3=pd.read_csv('/disk1/wangjialei/research/odir_main/stacking/test3.csv')[['ID','N','D','G','C','A','H','M','O']]
        data4=pd.read_csv('/disk1/wangjialei/research/odir_main/stacking/test4.csv')
        newdata=pd.merge(data1,data2,on='ID')
        newdata=pd.merge(newdata,data3,on='ID')
        newdata=pd.merge(newdata,data4,on='ID')
        return newdata

class second_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1=nn.Linear(in_features=32,out_features=64,bias=True)
        self.sigmoid=nn.Sigmoid()
        self.linear2=nn.Linear(in_features=64,out_features=128,bias=True)
        self.sigmoid=nn.Sigmoid()
        self.linear2=nn.Linear(in_features=64,out_features=128,bias=True)

        self.conv1=nn.Conv1d(in_channels=1,out_channels=2,kernel_size=3,stride=1)
        self.conv2=nn.Conv1d(in_channels=2,out_channels=4,kernel_size=7,stride=2)
        self.conv3=nn.Conv1d(in_channels=4,out_channels=2,kernel_size=9,stride=1)
        self.maxpooling=nn.AdaptiveMaxPool1d(output_size=32)
        self.linear3=nn.Linear(in_features=64,out_features=32,bias=True)
        self.linear4=nn.Linear(in_features=32,out_features=8)
    def forward(self,input):
        result=self.linear1(input)
        result=self.sigmoid(result)
        result=self.linear2(result)
        result=self.sigmoid(result)
        result=self.conv1(result)
        result=self.sigmoid(result)
        result=self.conv2(result)
        result=self.sigmoid(result)
        result=self.conv3(result)
        result=self.maxpooling(result)
        result=self.sigmoid(result)
        result=result.flatten(start_dim=1)
        result=self.linear3(result)
        result=self.sigmoid(result)
        result=self.linear4(result)
        return result
    def initialize_weights(self):
        for m in self.modules():
            # 判断是否属于Conv2d
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                # 判断是否有偏置
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data,0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1) 		 
                m.bias.data.zeros_()	

os.environ['CUDA_VISIBLE_DEVICES']='1'

def main():
    writer=SummaryWriter(f'stacking/second')
    dataset=Label_Dataset4(flatten=True,is_test=False)
    dataloader=DataLoader(dataset=dataset,batch_size=256)
    datasetvalid=Label_Dataset4(flatten=True,is_test=True)
    dataloader_valid=DataLoader(dataset=datasetvalid,batch_size=256)
    model=second_model().cuda()
    model.initialize_weights()
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.01)
    loss_func=nn.BCEWithLogitsLoss().cuda()
    lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30*len(dataloader),eta_min=0.000001,last_epoch=-1)
    for epoch in range(100):
        print(epoch)
        model.train()
        all_target=None
        all_output=None
        num_updates = epoch * len(dataloader)
        losses_m = AverageMeter()
        for i,(input,target) in enumerate(dataloader):
            input=input.cuda()
            target=target.cuda()
            output=model(input)
            loss=loss_func(output,target)
            if all_target is None:
                all_target = target.cpu().detach()
                all_output = output.cpu().detach().sigmoid()
            else:
                all_target = torch.cat((all_target, target.cpu().detach()), dim=0)
                all_output = torch.cat((all_output, output.cpu().detach().sigmoid()), dim=0)
            losses_m.update(loss.item(), input.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        cpu_target=all_target.numpy()
        cpu_output=all_output.numpy()
        cpu_output=np.where(cpu_output>=0.5,1,0)
        mAP = average_precision_score(cpu_target, cpu_output,average='micro')
        mAP_macro = average_precision_score(cpu_target, cpu_output,average='macro')
        mAP_weighted = average_precision_score(cpu_target, cpu_output,average='weighted')

        CP = precision_score(cpu_target, cpu_output, average='micro')
        OP = precision_score(cpu_target, cpu_output, average="macro")
        CR = recall_score(cpu_target, cpu_output, average='micro')
        OR = recall_score(cpu_target, cpu_output, average='macro')
        CF1 = f1_score(cpu_target, cpu_output, average='micro', pos_label=1)
        OF1 = f1_score(cpu_target, cpu_output, average='macro', pos_label=1)
        idx=epoch
        writer.add_scalar("loss",losses_m.avg,idx)
        writer.add_scalar("mAP micro",mAP,idx)
        writer.add_scalar("mAP macro",mAP_macro,idx)
        writer.add_scalar("mAP weighted",mAP_weighted,idx)

        writer.add_scalar("CP",CP,idx)
        writer.add_scalar("OP",OP,idx)
        writer.add_scalar("CR",CR,idx)
        writer.add_scalar("OR",OR,idx)
        writer.add_scalar("CF1",CF1,idx)
        writer.add_scalar("OF1",OF1,idx)
        model.eval()

        odir_pred=None
        odir_target=None  
        with torch.no_grad():
            for i,(input,target) in enumerate(dataloader_valid):
                input=input.cuda()
                target=target.cuda()
                output=model(input)
                loss=loss_func(output,target)
                if odir_pred is None:
                    odir_pred=output.sigmoid()
                    odir_target=target
                else:
                    odir_pred=torch.cat((odir_pred,output.sigmoid()),dim=0)
                    odir_target=torch.cat((odir_target,target),dim=0)
            odir_pred=odir_pred.detach().cpu().numpy()
            odir_target=odir_target.detach().cpu().numpy()
            kappa,f1,auc,_=ODIR_Metrics(odir_target,odir_pred)
            writer.add_scalar("valid_kappa",kappa,epoch)
            writer.add_scalar("valid_f1",f1,epoch)
            writer.add_scalar("valid_auc",auc,epoch)
if __name__=='__main__':
    main()