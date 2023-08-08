import torch
from torch import nn

class dice_loss(nn.Module):
    def __init__(self,p:float=0.9) -> None:
        super().__init__()
        self.p=p
    def forward(self,pred,target):
        pred=pred.sigmoid()
        pred=pred.contiguous().view(-1)
        target=target.contiguous().view(-1)
        intersection=(pred*target).sum()
        pred=pred.sum()
        target=target.sum()
        dice_coef=(2.*intersection+self.p)/(pred+target+self.p)
        return 1-dice_coef