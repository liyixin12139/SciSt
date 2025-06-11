import torchvision.models as models
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm import create_model
import typing
import warnings

class SelfAttention(nn.Module):
    def __init__(self,dim,num_heads,attn_drop_ratio=0.,proj_drop_ratio=0.):

        super(SelfAttention,self).__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim,3*dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    def forward(self,x):
        B,N,C = x.shape #x传入之前应该resize为B X C X dim
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = torch.matmul(q,k.transpose(-2,-1)) * self.scale
        attn = F.softmax(attn,dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn,v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,attn

class LayerNorm(nn.Module):
    def __init__(self,hidden_size,epsilon = 1e-12):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.epsilon = epsilon
    def forward(self,x):
        u = x.mean(-1,keepdim = True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.epsilon)
        return self.gamma * x + self.beta


class MLP(nn.Module):
    def __init__(self,hidden_size,output_num,dropout_rate = 0.):
        super(MLP,self).__init__()
        self.dense = nn.Linear(hidden_size,128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(128,output_num)
    def forward(self,x):
        feature = self.dropout(self.relu(self.dense(x)))
        return feature,F.softmax(self.classifier(feature))

class CNN_SASM(nn.Module):
    def __init__(self,model_name):
        super(CNN_SASM,self).__init__()
        if model_name == 'resnet18':
            cnn = models.resnet18(pretrained = True)
            self.cnn = nn.Sequential(*list(cnn.children())[:-2])
            dim = 49
            self.channel_selfattention = SelfAttention(dim,7,0.3,0.3)
            self.layernorm = LayerNorm(49)
            self.spatial_selfattention = SelfAttention(512,num_heads=8,attn_drop_ratio=0.3,proj_drop_ratio=0.3)
            self.layernorm2 = LayerNorm(512)
        elif model_name == 'resnet34':
            cnn = models.resnet34(pretrained = True)
            self.cnn = nn.Sequential(*list(cnn.children())[:-2])
            dim = 49
            self.channel_selfattention = SelfAttention(dim,7,0.3,0.3)
            self.layernorm = LayerNorm(49)
            self.spatial_selfattention = SelfAttention(512,num_heads=8,attn_drop_ratio=0.3,proj_drop_ratio=0.3)
            self.layernorm2 = LayerNorm(512)
        elif model_name == 'resnet50':
            cnn = models.resnet50(pretrained = True)
            self.cnn = nn.Sequential(*list(cnn.children())[:-2])
            dim = 49
            self.channel_selfattention = SelfAttention(dim,7,0.3,0.3)
            self.layernorm = LayerNorm(49)
            self.spatial_selfattention = SelfAttention(2048,num_heads=8,attn_drop_ratio=0.3,proj_drop_ratio=0.3)
            self.layernorm2 = LayerNorm(2048)
        elif  model_name == 'densenet121':
            cnn = models.densenet121(pretrained = True)
            self.cnn = nn.Sequential(*list(cnn.children())[:-1])
            dim = 49
            self.channel_selfattention = SelfAttention(dim, 7, 0.3, 0.3)
            self.layernorm = LayerNorm(49)
            self.spatial_selfattention = SelfAttention(1024,num_heads=8,attn_drop_ratio=0.3,proj_drop_ratio=0.3)
            self.layernorm2 = LayerNorm(1024)
        else:
            raise Exception('model is wrong! Only use resnet50 or densenet121')

    def forward(self,x,attn = False):
        x = self.cnn(x)
        B,C,w,h = x.shape
        x = x.reshape(B,C,w*h)
        residual_channel = x
        x_channel,attn_channel = self.channel_selfattention(x)
        x_channel += residual_channel
        x_channel = F.softmax(self.layernorm(x_channel),dim=-2)

        residual_spatial = x_channel.reshape(B,x_channel.shape[2],C)
        x_spatial = residual_spatial
        x_spatial,attn_spatial = self.spatial_selfattention(x_spatial)
        x_spatial += residual_spatial
        x_spatial = F.softmax(self.layernorm2(x_spatial),dim=-2)
        x = x_spatial.permute(0,2,1)
        img_feature = x.reshape(B,C,w,h)
        return img_feature
if __name__ == '__main__':
    model = CNN_SASM('resnet50')
    a = torch.randn(2,3,224,224)
    b = model(a)
    print(b.shape)
