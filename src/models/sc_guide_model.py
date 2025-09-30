import torch
import torch.nn as nn
from SingleCell_Ref.singleCell_models.exp_guide_module import Exp_Guide
from SingleCell_Ref.singleCell_models.exp_guide_module import MLPBlock
import torchvision.models as models
from SingleCell_Ref.singleCell_models.CNN_SASM import CNN_SASM


class SciSt(nn.Module):
    def __init__(self,img_encoder_name,gene_num):
        super().__init__()
        # Define three types of img_encoder
        if img_encoder_name=='resnet34': 
            self.img_encoder = CNN_SASM('resnet34')
            attention_dim = 512
        elif img_encoder_name=='resnet18':
            self.img_encoder = CNN_SASM('resnet18')
            attention_dim = 512
        elif img_encoder_name=='resnet50':
            self.img_encoder = CNN_SASM('resnet50')
            attention_dim = 2048
        else:
            attention_dim=None
            print('Img Encoder is invalid.')
        self.exp_guide_module = Exp_Guide(attention_dim,2,8,512,gene_num,3,1024) # Define the sc-guide module
        self.exp_encoder =  nn.Sequential(nn.Linear(gene_num,attention_dim), # Define the exp_encoder
                                          MLPBlock(attention_dim,attention_dim))
                                              
    def forward(self,img,orig_exp):
        b = img.shape[0] # Get Batch size
        img_emb = self.img_encoder(img)# img_encoder extracts image features -> (B, 512, 7, 7)
        exp_emb = self.exp_encoder(orig_exp).unsqueeze(1) # apply encoder to noise-exp
        exp_prediction = self.exp_guide_module(img_emb,exp_emb).view(b,-1)# Sc_guide module
        return exp_prediction

class SciSt_Ablation(nn.Module):
    '''
    Ablate the Sc-guide module by performing a simple concatenation
    '''
    def __init__(self,img_encoder_name,gene_num):
        super().__init__()
        if img_encoder_name=='resnet34':
            self.img_encoder = CNN_SASM('resnet34')
            attention_dim = 512
        elif img_encoder_name=='resnet18':
            self.img_encoder = CNN_SASM('resnet18')
            attention_dim = 512
        elif img_encoder_name=='resnet50':
            self.img_encoder = CNN_SASM('resnet50')
            attention_dim = 2048
        else:
            attention_dim=None
            print('Img Encoder is invalid.')
        self.exp_guide_module = Exp_Guide(attention_dim,2,8,512,gene_num,3,1024)
        self.exp_encoder =  nn.Sequential(nn.Linear(gene_num,attention_dim),
                                          MLPBlock(attention_dim,attention_dim))
                                              
        self.maxpool=nn.AdaptiveAvgPool2d(1)
        self.ln=nn.Linear(attention_dim*2,attention_dim*4)
        self.bn=nn.LayerNorm(attention_dim*4)
        self.ac1=nn.ReLU()
        self.final=nn.Linear(attention_dim*4,gene_num)
    def forward(self,img,orig_exp):
        b = img.shape[0]
        img_emb = self.img_encoder(img) #->(B,512,7,7)
        exp_emb = self.exp_encoder(orig_exp)
        img_emb=self.maxpool(img_emb).reshape(b,img_emb.shape[1])  #(B,512,7,7)->(B,512,1,1)
        exp_prediction=self.ac1(self.bn(self.ln(torch.concat((exp_emb,img_emb),dim=-1)))) #concat img feature(B,512)& exp feature(B,512)->(B,1024)
        return self.final(exp_prediction)



def get_model(model_name):
    model_dict = {
        'Sc_Guide_Model_with_TCGN':Sc_Guide_Model_with_TCGN,
        'Sc_Guide_Model_with_Resnet34':Sc_Guide_Model_with_Resnet34,
        'Sc_Guide_Model_with_Resnet50': Sc_Guide_Model_with_Resnet50,
        'SciSt':SciSt,
        'TCGN':TCGN,
        'SciSt_Ablation':SciSt_Ablation
    }
    return model_dict[model_name]



if __name__ == '__main__':
    model = SciSt('resnet34', 785)

