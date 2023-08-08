import torch
from torch import nn
import focal_loss
import dice_loss
class Fusion_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.focalloss=focal_loss.focal_loss()
        self.bce=nn.BCEWithLogitsLoss()
        self.diceloss=dice_loss.dice_loss()
    def forward(self,pred,target):
        loss=self.diceloss(pred,target)+self.bce(pred,target)
        return loss