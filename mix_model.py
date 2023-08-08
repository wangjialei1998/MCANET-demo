import timm
import torch
from torch import nn
class Mix_Models(nn.Module):
    def __init__(self,lock_base:bool,load_checkpoint:bool) -> None:
        super().__init__()
        self.model_resnet50d=timm.create_model("resnet50d",pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d=timm.create_model("res2net50d",pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_davit_base=timm.create_model("davit_base",pretrained=True,num_classes=0,drop_rate=0.1)#1024
        self.linear1=nn.Linear(in_features=2048,out_features=1024,bias=True)
        self.linear2=nn.Linear(in_features=2048,out_features=1024,bias=True)
        self.conv1d=nn.Conv1d(in_channels=3,out_channels=1,kernel_size=1,stride=1,bias=False)
        self.relu=nn.ReLU()
        self.linear3=nn.Linear(in_features=1024,out_features=512,bias=True)
        self.linear4=nn.Linear(in_features=512,out_features=256,bias=True)
        self.linear5=nn.Linear(in_features=256,out_features=8,bias=True)
        if lock_base:
            self.lock_base_model(self.model_resnet50d)
            self.lock_base_model(self.model_res2net50d)
            self.lock_base_model(self.model_davit_base)
        if load_checkpoint:
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_fold/train/20230704-120714-resnet50d-384/checkpoint-22.pth.tar',map_location='cpu')
            self.model_resnet50d.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_fold/train/20230704-112155-res2net50d-384/checkpoint-24.pth.tar',map_location='cpu')
            self.model_res2net50d.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_fold/train/20230704-095306-davit_base-384/checkpoint-28.pth.tar',map_location='cpu')
            self.model_davit_base.load_state_dict(state_dict["state_dict"],strict=False)
    def forward(self,input):
        result1=self.model_resnet50d(input)
        result1=self.linear1(result1)
        result2=self.model_res2net50d(input)
        result2=self.linear2(result2)
        result3=self.model_davit_base(input)
        result=torch.stack((result1,result2,result3),dim=1)#3,1024
        result=self.conv1d(result)
        result=self.relu(result)
        result=self.linear3(result)
        result=self.relu(result)
        result=self.linear4(result)
        result=self.relu(result)
        result=self.linear5(result)
        return torch.squeeze(result)
    def lock_base_model(self,model):
        for param in model.parameters():
            param.requires_grad = False
    


# class ChannelAttention(nn.Module):

#     def __init__(self, dim, num_heads=8, qkv_bias=False):
#         super().__init__()
#         self.num_heads = num_heads # 这里的num_heads实际上是num_groups
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim)

#     def forward(self, x):
#         B, N, C = x.shape
#         # 得到query，key和value，是在channel维度上进行线性投射
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         k = k * self.scale
#         attention = k.transpose(-1, -2) @ v # 对维度进行反转
#         attention = attention.softmax(dim=-1)
#         x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         return x
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class resnet50d_without_last_two(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model=nn.Sequential(*list(timm.create_model('resnet50d').children())[:-2])
    def forward(self,x):
        return self.model(x)
class Mix_Models_with_Channel_Attention(nn.Module):
    def __init__(self,lock_base:bool,load_checkpoint:bool) -> None:
        super().__init__()
        self.model_resnet50d=timm.create_model('resnet50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_davit_base=timm.create_model('davit_base',pretrained=True,num_classes=0,drop_rate=0.1)#1024
        if lock_base:
            self.lock_base_model(self.model_resnet50d)
            self.lock_base_model(self.model_res2net50d)
            self.lock_base_model(self.model_davit_base)
        if load_checkpoint:
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_fold/train/20230704-120714-resnet50d-384/checkpoint-22.pth.tar',map_location='cpu')
            self.model_resnet50d.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_fold/train/20230704-112155-res2net50d-384/checkpoint-24.pth.tar',map_location='cpu')
            self.model_res2net50d.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_fold/train/20230704-095306-davit_base-384/checkpoint-28.pth.tar',map_location='cpu')
            self.model_davit_base.load_state_dict(state_dict["state_dict"],strict=False)
        self.model_resnet50d=nn.Sequential(*list(self.model_resnet50d.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d=nn.Sequential(*list(self.model_res2net50d.children())[:-2])#[1, 2048, 16, 16]
        self.model_davit_base=nn.Sequential(*list(self.model_davit_base.children())[:-2])#[1, 1024, 16, 16]
        self.relu=nn.ReLU()
        self.linear3=nn.Linear(in_features=5120,out_features=2048,bias=True)
        self.linear4=nn.Linear(in_features=2048,out_features=512,bias=True)
        self.linear5=nn.Linear(in_features=512,out_features=8,bias=True)
        self.channelattention=ChannelAttention(in_planes=5120)
    def forward(self,input):
        result1=self.model_resnet50d(input)
        result2=self.model_res2net50d(input)
        result3=self.model_davit_base(input)
        result=torch.cat((result1,result2,result3),dim=1)#3,1024
        result=self.channelattention(result)
        result=self.relu(result)
        result=result.squeeze()
        result=self.linear3(result)
        result=self.relu(result)
        result=self.linear4(result)
        result=self.relu(result)
        result=self.linear5(result)
        return torch.squeeze(result)
    def lock_base_model(self,model):
        for param in model.parameters():
            param.requires_grad = False


class Mix_Models_with_Channel_Attention_8(nn.Module):
    def __init__(self,lock_base:bool,load_checkpoint:bool) -> None:
        super().__init__()
        self.model_res2net50d1=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d2=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d3=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d4=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d5=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d6=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d7=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d8=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        if lock_base:
            self.lock_base_model(self.model_res2net50d1)
            self.lock_base_model(self.model_res2net50d2)
            self.lock_base_model(self.model_res2net50d3)
            self.lock_base_model(self.model_res2net50d4)
            self.lock_base_model(self.model_res2net50d5)
            self.lock_base_model(self.model_res2net50d6)
            self.lock_base_model(self.model_res2net50d7)
            self.lock_base_model(self.model_res2net50d8)
        if load_checkpoint:
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-213105-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d1.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-220750-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d2.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-224444-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d3.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-232126-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d4.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-235821-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d5.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230705-003534-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d6.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230705-011314-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d7.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230705-015044-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d8.load_state_dict(state_dict["state_dict"],strict=False)
        self.model_res2net50d1=nn.Sequential(*list(self.model_res2net50d1.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d2=nn.Sequential(*list(self.model_res2net50d2.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d3=nn.Sequential(*list(self.model_res2net50d3.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d4=nn.Sequential(*list(self.model_res2net50d4.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d5=nn.Sequential(*list(self.model_res2net50d5.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d6=nn.Sequential(*list(self.model_res2net50d6.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d7=nn.Sequential(*list(self.model_res2net50d7.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d8=nn.Sequential(*list(self.model_res2net50d8.children())[:-2])#[1, 2048, 16, 16]

        self.relu=nn.ReLU()
        self.linear3=nn.Linear(in_features=16384,out_features=8192,bias=True)
        self.linear4=nn.Linear(in_features=8192,out_features=2048,bias=True)
        self.linear5=nn.Linear(in_features=2048,out_features=512,bias=True)
        self.linear6=nn.Linear(in_features=512,out_features=8,bias=True)
        self.channelattention=ChannelAttention(in_planes=16384)
    def forward(self,input):
        result1=self.model_res2net50d1(input)
        result2=self.model_res2net50d2(input)
        result3=self.model_res2net50d3(input)
        result4=self.model_res2net50d4(input)
        result5=self.model_res2net50d5(input)
        result6=self.model_res2net50d6(input)
        result7=self.model_res2net50d7(input)
        result8=self.model_res2net50d8(input)

        result=torch.cat((result1,result2,result3,result4,result5,result6,result7,result8),dim=1)#3,1024
        result=self.channelattention(result)
        result=self.relu(result)
        result=result.squeeze()
        result=self.linear3(result)
        result=self.relu(result)
        result=self.linear4(result)
        result=self.relu(result)
        result=self.linear5(result)
        result=self.relu(result)
        result=self.linear6(result)
        return torch.squeeze(result)
    def lock_base_model(self,model):
        for param in model.parameters():
            param.requires_grad = False

class Mix_Models_with_Channel_Attention_4(nn.Module):
    def __init__(self,lock_base:bool,load_checkpoint:bool) -> None:
        super().__init__()
        self.model_res2net50d1=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d2=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d3=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d4=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        if lock_base:
            self.lock_base_model(self.model_res2net50d1)
            self.lock_base_model(self.model_res2net50d2)
            self.lock_base_model(self.model_res2net50d3)
            self.lock_base_model(self.model_res2net50d4)
        if load_checkpoint:
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight4_fold/train/20230705-111232-res2net50d-512/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d1.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight4_fold/train/20230705-115954-res2net50d-512/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d2.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight4_fold/train/20230705-124705-res2net50d-512/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d3.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight4_fold/train/20230705-133425-res2net50d-512/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d4.load_state_dict(state_dict["state_dict"],strict=False)
        self.model_res2net50d1=nn.Sequential(*list(self.model_res2net50d1.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d2=nn.Sequential(*list(self.model_res2net50d2.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d3=nn.Sequential(*list(self.model_res2net50d3.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d4=nn.Sequential(*list(self.model_res2net50d4.children())[:-2])#[1, 2048, 16, 16]


        self.relu=nn.ReLU()
        self.linear3=nn.Linear(in_features=8192,out_features=4096,bias=True)
        self.linear4=nn.Linear(in_features=4096,out_features=2048,bias=True)
        self.linear5=nn.Linear(in_features=2048,out_features=512,bias=True)
        self.linear6=nn.Linear(in_features=512,out_features=8,bias=True)
        self.channelattention=ChannelAttention(in_planes=8192)
    def forward(self,input):
        result1=self.model_res2net50d1(input)
        result2=self.model_res2net50d2(input)
        result3=self.model_res2net50d3(input)
        result4=self.model_res2net50d4(input)

        result=torch.cat((result1,result2,result3,result4),dim=1)#3,1024
        result=self.channelattention(result)
        result=self.relu(result)
        result=result.squeeze()
        result=self.linear3(result)
        result=self.relu(result)
        result=self.linear4(result)
        result=self.relu(result)
        result=self.linear5(result)
        result=self.relu(result)
        result=self.linear6(result)
        return torch.squeeze(result)
    def lock_base_model(self,model):
        for param in model.parameters():
            param.requires_grad = False
class Mix_Models_with_Channel_Attention_4_right(nn.Module):
    def __init__(self,lock_base:bool,load_checkpoint:bool) -> None:
        super().__init__()
        self.model_res2net50d1=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d2=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d3=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d4=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        if lock_base:
            self.lock_base_model(self.model_res2net50d1)
            self.lock_base_model(self.model_res2net50d2)
            self.lock_base_model(self.model_res2net50d3)
            self.lock_base_model(self.model_res2net50d4)
        if load_checkpoint:
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight4_fold/train/20230705-111232-res2net50d-512/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d1.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight4_fold/train/20230705-115954-res2net50d-512/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d2.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight4_fold/train/20230705-124705-res2net50d-512/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d3.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight4_fold/train/20230705-133425-res2net50d-512/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d4.load_state_dict(state_dict["state_dict"],strict=False)
        self.model_res2net50d1=nn.Sequential(*list(self.model_res2net50d1.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d2=nn.Sequential(*list(self.model_res2net50d2.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d3=nn.Sequential(*list(self.model_res2net50d3.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d4=nn.Sequential(*list(self.model_res2net50d4.children())[:-2])#[1, 2048, 16, 16]
        self.maxpooling=nn.MaxPool2d(kernel_size=(12,12),stride=1)
        self.relu=nn.ReLU()
        self.linear3=nn.Linear(in_features=8192,out_features=4096,bias=True)
        self.linear4=nn.Linear(in_features=4096,out_features=2048,bias=True)
        self.linear5=nn.Linear(in_features=2048,out_features=512,bias=True)
        self.linear6=nn.Linear(in_features=512,out_features=8,bias=True)
        self.channelattention=ChannelAttention(in_planes=8192)
    def forward(self,input):
        result1=self.model_res2net50d1(input)
        result2=self.model_res2net50d2(input)
        result3=self.model_res2net50d3(input)
        result4=self.model_res2net50d4(input)
        result=torch.cat((result1,result2,result3,result4),dim=1)#3,1024
        channel_result=self.channelattention(result)
        result=result*channel_result
        result=self.maxpooling(result)
        result=self.relu(result)
        result=result.squeeze()
        result=self.linear3(result)
        result=self.relu(result)
        result=self.linear4(result)
        result=self.relu(result)
        result=self.linear5(result)
        result=self.relu(result)
        result=self.linear6(result)
        return torch.squeeze(result)
    def lock_base_model(self,model):
        for param in model.parameters():
            param.requires_grad = False
class Mix_Models_with_Channel_Attention_2(nn.Module):
    def __init__(self,lock_base:bool,load_checkpoint:bool) -> None:
        super().__init__()
        self.model_res2net50d1=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d2=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        if lock_base:
            self.lock_base_model(self.model_res2net50d1)
            self.lock_base_model(self.model_res2net50d2)
        if load_checkpoint:
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-213105-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d1.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-220750-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d2.load_state_dict(state_dict["state_dict"],strict=False)
        self.model_res2net50d1=nn.Sequential(*list(self.model_res2net50d1.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d2=nn.Sequential(*list(self.model_res2net50d2.children())[:-2])#[1, 2048, 16, 16]

        self.relu=nn.ReLU()
        self.linear3=nn.Linear(in_features=4096,out_features=2048,bias=True)
        self.linear4=nn.Linear(in_features=2048,out_features=1024,bias=True)
        self.linear5=nn.Linear(in_features=1024,out_features=512,bias=True)
        self.linear6=nn.Linear(in_features=512,out_features=8,bias=True)
        self.channelattention=ChannelAttention(in_planes=4096)
    def forward(self,input):
        result1=self.model_res2net50d1(input)
        result2=self.model_res2net50d2(input)

        result=torch.cat((result1,result2),dim=1)#3,1024
        result=self.channelattention(result)
        result=self.relu(result)
        result=result.squeeze()
        result=self.linear3(result)
        result=self.relu(result)
        result=self.linear4(result)
        result=self.relu(result)
        result=self.linear5(result)
        result=self.relu(result)
        result=self.linear6(result)
        return torch.squeeze(result)
    def lock_base_model(self,model):
        for param in model.parameters():
            param.requires_grad = False

class Mix_Models_with_Channel_Attention_2_right(nn.Module):
    def __init__(self,lock_base:bool,load_checkpoint:bool) -> None:
        super().__init__()
        self.model_res2net50d1=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d2=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        if lock_base:
            self.lock_base_model(self.model_res2net50d1)
            self.lock_base_model(self.model_res2net50d2)
        if load_checkpoint:
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-213105-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d1.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-220750-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d2.load_state_dict(state_dict["state_dict"],strict=False)
        self.model_res2net50d1=nn.Sequential(*list(self.model_res2net50d1.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d2=nn.Sequential(*list(self.model_res2net50d2.children())[:-2])#[1, 2048, 16, 16]
        self.maxpooling=nn.MaxPool2d(kernel_size=(12,12),stride=1)
        self.relu=nn.ReLU()
        self.linear3=nn.Linear(in_features=4096,out_features=2048,bias=True)
        self.linear4=nn.Linear(in_features=2048,out_features=1024,bias=True)
        self.linear5=nn.Linear(in_features=1024,out_features=512,bias=True)
        self.linear6=nn.Linear(in_features=512,out_features=8,bias=True)
        self.channelattention=ChannelAttention(in_planes=4096)
    def forward(self,input):
        result1=self.model_res2net50d1(input)
        result2=self.model_res2net50d2(input)
        result=torch.cat((result1,result2),dim=1)#3,1024
        channel_result=self.channelattention(result)
        result=result*channel_result
        result=self.maxpooling(result)
        result=self.relu(result)
        result=result.squeeze()
        result=self.linear3(result)
        result=self.relu(result)
        result=self.linear4(result)
        result=self.relu(result)
        result=self.linear5(result)
        result=self.relu(result)
        result=self.linear6(result)
        return torch.squeeze(result)
    def lock_base_model(self,model):
        for param in model.parameters():
            param.requires_grad = False




class Mix_Models_with_Channel_Attention_2(nn.Module):
    def __init__(self,lock_base:bool,load_checkpoint:bool) -> None:
        super().__init__()
        self.model_res2net50d1=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d2=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        if lock_base:
            self.lock_base_model(self.model_res2net50d1)
            self.lock_base_model(self.model_res2net50d2)
        if load_checkpoint:
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-213105-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d1.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-220750-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d2.load_state_dict(state_dict["state_dict"],strict=False)
        self.model_res2net50d1=nn.Sequential(*list(self.model_res2net50d1.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d2=nn.Sequential(*list(self.model_res2net50d2.children())[:-2])#[1, 2048, 16, 16]

        self.relu=nn.ReLU()
        self.linear3=nn.Linear(in_features=4096,out_features=2048,bias=True)
        self.linear4=nn.Linear(in_features=2048,out_features=1024,bias=True)
        self.linear5=nn.Linear(in_features=1024,out_features=512,bias=True)
        self.linear6=nn.Linear(in_features=512,out_features=8,bias=True)
        self.channelattention=ChannelAttention(in_planes=4096)
    def forward(self,input):
        result1=self.model_res2net50d1(input)
        result2=self.model_res2net50d2(input)

        result=torch.cat((result1,result2),dim=1)#3,1024
        result=self.channelattention(result)
        result=self.relu(result)
        result=result.squeeze()
        result=self.linear3(result)
        result=self.relu(result)
        result=self.linear4(result)
        result=self.relu(result)
        result=self.linear5(result)
        result=self.relu(result)
        result=self.linear6(result)
        return torch.squeeze(result)
    def lock_base_model(self,model):
        for param in model.parameters():
            param.requires_grad = False

class Mix_Models_with_Channel_Attention_2_right_final(nn.Module):
    def __init__(self,lock_base:bool,load_checkpoint:bool) -> None:
        super().__init__()
        self.model_res2net50d1=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d2=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        if lock_base:
            self.lock_base_model(self.model_res2net50d1)
            self.lock_base_model(self.model_res2net50d2)
        if load_checkpoint:
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-213105-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d1.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-220750-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d2.load_state_dict(state_dict["state_dict"],strict=False)
        self.model_res2net50d1=nn.Sequential(*list(self.model_res2net50d1.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d2=nn.Sequential(*list(self.model_res2net50d2.children())[:-2])#[1, 2048, 16, 16]
        self.maxpooling=nn.MaxPool2d(kernel_size=(12,12),stride=1)
        self.relu=nn.ReLU()
        self.linear3=nn.Linear(in_features=2048,out_features=1024,bias=True)
        self.linear4=nn.Linear(in_features=1024,out_features=512,bias=True)
        self.linear5=nn.Linear(in_features=512,out_features=256,bias=True)
        self.linear6=nn.Linear(in_features=256,out_features=8,bias=True)
        self.channelattention=ChannelAttention(in_planes=2048)
        self.channelattention1=ChannelAttention(in_planes=2048)
        self.channelattention2=ChannelAttention(in_planes=2048)

    def forward(self,input):
        result1=self.model_res2net50d1(input)
        result2=self.model_res2net50d2(input)
        result1_channel=self.channelattention1(result1)
        result2_channel=self.channelattention2(result2)
        result1=result1*result1_channel
        result2=result2*result2_channel



        # result=torch.cat((result1,result2),dim=1)#3,1024
        result=result1+result2
        channel_result=self.channelattention(result)
        result=result*channel_result

        result=result+result1+result2

        result=self.maxpooling(result)
        result=self.relu(result)
        result=result.squeeze()
        result=self.linear3(result)
        result=self.relu(result)
        result=self.linear4(result)
        result=self.relu(result)
        result=self.linear5(result)
        result=self.relu(result)
        result=self.linear6(result)
        return torch.squeeze(result)
    def lock_base_model(self,model):
        for param in model.parameters():
            param.requires_grad = False



class Mix_Models_with_Channel_Attention_3_right_final(nn.Module):
    def __init__(self,lock_base:bool,load_checkpoint:bool) -> None:
        super().__init__()
        self.model_res2net50d1=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d2=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d3=timm.create_model('res2net50d',pretrained=True,num_classes=0,drop_rate=0.1)#2048

        if lock_base:
            self.lock_base_model(self.model_res2net50d1)
            self.lock_base_model(self.model_res2net50d2)
            self.lock_base_model(self.model_res2net50d3)

        if load_checkpoint:
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-213105-res2net50d-384/checkpoint-29.pth.tar',map_location='cpu')
            self.model_res2net50d1.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-220750-res2net50d-384/checkpoint-20.pth.tar',map_location='cpu')
            self.model_res2net50d2.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load('/disk1/wangjialei/research/odir_main/all_single_weight/train/20230704-220750-res2net50d-384/checkpoint-24.pth.tar',map_location='cpu')
            self.model_res2net50d3.load_state_dict(state_dict["state_dict"],strict=False)
        self.model_res2net50d1=nn.Sequential(*list(self.model_res2net50d1.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d2=nn.Sequential(*list(self.model_res2net50d2.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d3=nn.Sequential(*list(self.model_res2net50d3.children())[:-2])#[1, 2048, 16, 16]
        self.downsample1=nn.Conv2d(in_channels=2048,out_channels=1024,kernel_size=(1,1),stride=1)
        self.downsample2=nn.Conv2d(in_channels=2048,out_channels=1024,kernel_size=(1,1),stride=1)
        self.downsample3=nn.Conv2d(in_channels=2048,out_channels=1024,kernel_size=(1,1),stride=1)

        self.maxpooling=nn.MaxPool2d(kernel_size=(12,12),stride=1)
        self.relu=nn.ReLU()
        self.linear3=nn.Linear(in_features=1024,out_features=512,bias=True)
        self.linear4=nn.Linear(in_features=512,out_features=256,bias=True)
        self.linear5=nn.Linear(in_features=256,out_features=128,bias=True)
        self.linear6=nn.Linear(in_features=128,out_features=8,bias=True)
        self.channelattention=ChannelAttention(in_planes=1024)
        self.channelattention1=ChannelAttention(in_planes=1024)
        self.channelattention2=ChannelAttention(in_planes=1024)
        self.channelattention3=ChannelAttention(in_planes=1024)

    def forward(self,input):
        result1=self.model_res2net50d1(input)
        result2=self.model_res2net50d2(input)
        result3=self.model_res2net50d3(input)
        result1=self.downsample1(result1)
        result2=self.downsample2(result2)
        result3=self.downsample3(result3)
        result1_channel=self.channelattention1(result1)
        result2_channel=self.channelattention2(result2)
        result3_channel=self.channelattention3(result3)
        result1=result1*result1_channel
        result2=result2*result2_channel
        result3=result3*result3_channel



        # result=torch.cat((result1,result2),dim=1)#3,1024
        result=result1+result2+result3
        channel_result=self.channelattention(result)
        result=result*channel_result

        result=result+result1+result2+result3

        result=self.maxpooling(result)
        result=self.relu(result)
        result=result.squeeze()
        result=self.linear3(result)
        result=self.relu(result)
        result=self.linear4(result)
        result=self.relu(result)
        result=self.linear5(result)
        result=self.relu(result)
        result=self.linear6(result)
        return torch.squeeze(result)
    def lock_base_model(self,model):
        for param in model.parameters():
            param.requires_grad = False



class Mix_Models_with_Channel_Attention_3_right_for_all_model(nn.Module):
    def __init__(self,lock_base:bool,load_checkpoint:bool,model_name:str,model_checkpoint:list=None) -> None:
        super().__init__()
        self.model_res2net50d1=timm.create_model(model_name,pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d2=timm.create_model(model_name,pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d3=timm.create_model(model_name,pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_name=model_name
        if lock_base:
            self.lock_base_model(self.model_res2net50d1)
            self.lock_base_model(self.model_res2net50d2)
            self.lock_base_model(self.model_res2net50d3)

        if load_checkpoint:
            state_dict=torch.load(model_checkpoint[0],map_location='cpu')
            self.model_res2net50d1.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load(model_checkpoint[1],map_location='cpu')
            self.model_res2net50d2.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load(model_checkpoint[2],map_location='cpu')
            self.model_res2net50d3.load_state_dict(state_dict["state_dict"],strict=False)
        self.model_res2net50d1=nn.Sequential(*list(self.model_res2net50d1.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d2=nn.Sequential(*list(self.model_res2net50d2.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d3=nn.Sequential(*list(self.model_res2net50d3.children())[:-2])#[1, 2048, 16, 16]
        backbone_shape=self.model_res2net50d1(torch.ones(1,3,384,384)).shape
        if self.model_name=='swin_base_patch4_window12_384':
            self.downsample1=nn.Conv2d(in_channels=backbone_shape[3],out_channels=1024,kernel_size=(1,1),stride=1)
            self.downsample2=nn.Conv2d(in_channels=backbone_shape[3],out_channels=1024,kernel_size=(1,1),stride=1)
            self.downsample3=nn.Conv2d(in_channels=backbone_shape[3],out_channels=1024,kernel_size=(1,1),stride=1)
        else:
            self.downsample1=nn.Conv2d(in_channels=backbone_shape[1],out_channels=1024,kernel_size=(1,1),stride=1)
            self.downsample2=nn.Conv2d(in_channels=backbone_shape[1],out_channels=1024,kernel_size=(1,1),stride=1)
            self.downsample3=nn.Conv2d(in_channels=backbone_shape[1],out_channels=1024,kernel_size=(1,1),stride=1)
        if len(backbone_shape)!=4:
            self.maxpooling=nn.MaxPool2d(kernel_size=(1,1),stride=1)
        elif self.model_name=='swin_base_patch4_window12_384':
            self.maxpooling=nn.MaxPool2d(kernel_size=(backbone_shape[1],backbone_shape[2]),stride=1)
        else:
            self.maxpooling=nn.MaxPool2d(kernel_size=(backbone_shape[2],backbone_shape[3]),stride=1)
        self.relu=nn.ReLU()
        self.linear3=nn.Linear(in_features=1024,out_features=512,bias=True)
        self.linear4=nn.Linear(in_features=512,out_features=256,bias=True)
        self.linear5=nn.Linear(in_features=256,out_features=128,bias=True)
        self.linear6=nn.Linear(in_features=128,out_features=8,bias=True)
        self.channelattention=ChannelAttention(in_planes=1024)
        self.channelattention1=ChannelAttention(in_planes=1024)
        self.channelattention2=ChannelAttention(in_planes=1024)
        self.channelattention3=ChannelAttention(in_planes=1024)

    def forward(self,input):
        result1=self.model_res2net50d1(input)
        result2=self.model_res2net50d2(input)
        result3=self.model_res2net50d3(input)
        if self.model_name=='swin_base_patch4_window12_384':
            result1=result1.permute(0,3,1,2)
            result2=result2.permute(0,3,1,2)
            result3=result3.permute(0,3,1,2)
        if self.model_name=='densenet121':
            result1=result1.unsqueeze(-1).unsqueeze(-1)
            result2=result2.unsqueeze(-1).unsqueeze(-1)
            result3=result3.unsqueeze(-1).unsqueeze(-1)

        result1=self.downsample1(result1)
        result2=self.downsample2(result2)
        result3=self.downsample3(result3)
        result1_channel=self.channelattention1(result1)
        result2_channel=self.channelattention2(result2)
        result3_channel=self.channelattention3(result3)
        result1=result1*result1_channel
        result2=result2*result2_channel
        result3=result3*result3_channel



        # result=torch.cat((result1,result2),dim=1)#3,1024
        result=result1+result2+result3
        channel_result=self.channelattention(result)
        result=result*channel_result

        result=result+result1+result2+result3

        result=self.maxpooling(result)
        result=self.relu(result)
        result=result.squeeze()
        result=self.linear3(result)
        result=self.relu(result)
        result=self.linear4(result)
        result=self.relu(result)
        result=self.linear5(result)
        result=self.relu(result)
        result=self.linear6(result)
        return torch.squeeze(result)
    def lock_base_model(self,model):
        for param in model.parameters():
            param.requires_grad = False




class Mix_Models_with_Channel_Attention_3_right_for_all_model_no_fusion(nn.Module):
    def __init__(self,lock_base:bool,load_checkpoint:bool,model_name:str,model_checkpoint:list=None) -> None:
        super().__init__()
        self.model_res2net50d1=timm.create_model(model_name,pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d2=timm.create_model(model_name,pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_res2net50d3=timm.create_model(model_name,pretrained=True,num_classes=0,drop_rate=0.1)#2048
        self.model_name=model_name
        if lock_base:
            self.lock_base_model(self.model_res2net50d1)
            self.lock_base_model(self.model_res2net50d2)
            self.lock_base_model(self.model_res2net50d3)

        if load_checkpoint:
            state_dict=torch.load(model_checkpoint[0],map_location='cpu')
            self.model_res2net50d1.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load(model_checkpoint[1],map_location='cpu')
            self.model_res2net50d2.load_state_dict(state_dict["state_dict"],strict=False)
            state_dict=torch.load(model_checkpoint[2],map_location='cpu')
            self.model_res2net50d3.load_state_dict(state_dict["state_dict"],strict=False)
        self.model_res2net50d1=nn.Sequential(*list(self.model_res2net50d1.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d2=nn.Sequential(*list(self.model_res2net50d2.children())[:-2])#[1, 2048, 16, 16]
        self.model_res2net50d3=nn.Sequential(*list(self.model_res2net50d3.children())[:-2])#[1, 2048, 16, 16]
        backbone_shape=self.model_res2net50d1(torch.ones(1,3,384,384)).shape
        if self.model_name=='swin_base_patch4_window12_384':
            self.downsample1=nn.Conv2d(in_channels=backbone_shape[3],out_channels=1024,kernel_size=(1,1),stride=1)
            self.downsample2=nn.Conv2d(in_channels=backbone_shape[3],out_channels=1024,kernel_size=(1,1),stride=1)
            self.downsample3=nn.Conv2d(in_channels=backbone_shape[3],out_channels=1024,kernel_size=(1,1),stride=1)
        else:
            self.downsample1=nn.Conv2d(in_channels=backbone_shape[1],out_channels=1024,kernel_size=(1,1),stride=1)
            self.downsample2=nn.Conv2d(in_channels=backbone_shape[1],out_channels=1024,kernel_size=(1,1),stride=1)
            self.downsample3=nn.Conv2d(in_channels=backbone_shape[1],out_channels=1024,kernel_size=(1,1),stride=1)
        if len(backbone_shape)!=4:
            self.maxpooling=nn.MaxPool2d(kernel_size=(1,1),stride=1)
        elif self.model_name=='swin_base_patch4_window12_384':
            self.maxpooling=nn.MaxPool2d(kernel_size=(backbone_shape[1],backbone_shape[2]),stride=1)
        else:
            self.maxpooling=nn.MaxPool2d(kernel_size=(backbone_shape[2],backbone_shape[3]),stride=1)
        self.relu=nn.ReLU()
        self.linear3=nn.Linear(in_features=1024,out_features=512,bias=True)
        self.linear4=nn.Linear(in_features=512,out_features=256,bias=True)
        self.linear5=nn.Linear(in_features=256,out_features=128,bias=True)
        self.linear6=nn.Linear(in_features=128,out_features=8,bias=True)

    def forward(self,input):
        result1=self.model_res2net50d1(input)
        result2=self.model_res2net50d2(input)
        result3=self.model_res2net50d3(input)
        if self.model_name=='swin_base_patch4_window12_384':
            result1=result1.permute(0,3,1,2)
            result2=result2.permute(0,3,1,2)
            result3=result3.permute(0,3,1,2)
        if self.model_name=='densenet121':
            result1=result1.unsqueeze(-1).unsqueeze(-1)
            result2=result2.unsqueeze(-1).unsqueeze(-1)
            result3=result3.unsqueeze(-1).unsqueeze(-1)

        result1=self.downsample1(result1)
        result2=self.downsample2(result2)
        result3=self.downsample3(result3)
        # result=torch.cat((result1,result2),dim=1)#3,1024
        result=result1+result2+result3
        result=self.maxpooling(result)
        result=self.relu(result)
        result=result.squeeze()
        result=self.linear3(result)
        result=self.relu(result)
        result=self.linear4(result)
        result=self.relu(result)
        result=self.linear5(result)
        result=self.relu(result)
        result=self.linear6(result)
        return torch.squeeze(result)
    def lock_base_model(self,model):
        for param in model.parameters():
            param.requires_grad = False
# model=Mix_Models_with_Channel_Attention_4_right(lock_base=True,load_checkpoint=True)
# print(model(torch.ones(16,3,384,384)).shape)