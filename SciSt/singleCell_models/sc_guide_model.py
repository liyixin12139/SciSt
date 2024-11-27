from paperTCGN.TCGN_model.TCGN import TCGN
import torch
import torch.nn as nn
from SingleCell_Ref.singleCell_models.exp_guide_module import Exp_Guide
from SingleCell_Ref.singleCell_models.exp_guide_module import MLPBlock
import torchvision.models as models
from SingleCell_Ref.singleCell_models.CNN_SASM import CNN_SASM

class Sc_Guide_Model_with_TCGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcgn = TCGN()
        self.exp_guide_module = Exp_Guide(368,2,8,512,785,3,1024)
        self.exp_encoder =  nn.Sequential(nn.Linear(785,368),
                                          MLPBlock(368,512))
                                            
    def forward(self,img,orig_exp):
        tcgn_out,img_emb = self.tcgn(img)
        exp_emb = self.exp_encoder(orig_exp).unsqueeze(1)
        exp_prediction = self.exp_guide_module(img_emb,exp_emb)
        return exp_prediction

class Sc_Guide_Model_with_Resnet34(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = models.resnet34(pretrained=True)
        self.img_encoder = nn.Sequential(*list(encoder.children())[:-2])  #->(B,512,7,7)
        self.exp_guide_module = Exp_Guide(512,2,8,512,785,3,1024)
        self.exp_encoder =  nn.Sequential(nn.Linear(785,512),
                                          MLPBlock(512,512))
                                             
    def forward(self,img,orig_exp):
        img_emb = self.img_encoder(img) #->(B,512,7,7)
        exp_emb = self.exp_encoder(orig_exp).unsqueeze(1)
        exp_prediction = self.exp_guide_module(img_emb,exp_emb)
        return exp_prediction

class Sc_Guide_Model_with_Resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = models.resnet50(pretrained=True)
        self.img_encoder = nn.Sequential(*list(encoder.children())[:-2])  #->(B,512,7,7)
        self.exp_guide_module = Exp_Guide(2048,2,8,512,785,3,1024)
        self.exp_encoder =  nn.Sequential(nn.Linear(785,2048),
                                          MLPBlock(2048,512))
                                            
    def forward(self,img,orig_exp):
        img_emb = self.img_encoder(img) #->(B,2048,7,7)
        exp_emb = self.exp_encoder(orig_exp).unsqueeze(1)
        exp_prediction = self.exp_guide_module(img_emb,exp_emb)
        return exp_prediction

class Sc_Guide_Model_with_CNNSASM(nn.Module):
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
                                              
    def forward(self,img,orig_exp):
        b = img.shape[0]
        img_emb = self.img_encoder(img) #->(B,2048,7,7)
        exp_emb = self.exp_encoder(orig_exp).unsqueeze(1)
        exp_prediction = self.exp_guide_module(img_emb,exp_emb).view(b,-1)
        return exp_prediction

class Concat_Model_for_Ablation(nn.Module):
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
        img_emb = self.img_encoder(img) #->(B,2048,7,7)
        exp_emb = self.exp_encoder(orig_exp)
        img_emb=self.maxpool(img_emb).reshape(b,img_emb.shape[1])  #(B,512,7,7)->(B,512,1,1)
        exp_prediction=self.ac1(self.bn(self.ln(torch.concat((exp_emb,img_emb),dim=-1))))
        return self.final(exp_prediction)



def get_model(model_name):
    model_dict = {
        'Sc_Guide_Model_with_TCGN':Sc_Guide_Model_with_TCGN,
        'Sc_Guide_Model_with_Resnet34':Sc_Guide_Model_with_Resnet34,
        'Sc_Guide_Model_with_Resnet50': Sc_Guide_Model_with_Resnet50,
        'Sc_Guide_Model_with_CNN_SASM':Sc_Guide_Model_with_CNNSASM,
        'TCGN':TCGN,
        'Concat_Model_for_Ablation':Concat_Model_for_Ablation
    }
    return model_dict[model_name]



if __name__ == '__main__':
    model = Sc_Guide_Model_with_CNNSASM('resnet34', 785)

