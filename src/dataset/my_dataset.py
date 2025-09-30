import os
import json
import torchvision.transforms as transforms
from PIL import ImageFile, Image
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from SingleCell_Ref.dataset_utils.create_orig_exp import create_orig_exp_for_patch,her_create_orig_exp_for_patch_scp
class Initiate_dataset(Dataset):
    def __init__(self,patch_path,exp_label_path,orig_path,mode,fold,data_name):
        super(Initiate_dataset,self).__init__()
        self.patch_path = patch_path
        self.label_path = exp_label_path
        self.orig_path = orig_path
        self.mode = mode
        self.wsi_ids_lst = [i.split('.')[0] for i in os.listdir(exp_label_path)]
        self.wsi_ids_lst.sort()
        self.train_transform = transforms.Compose(
            [transforms.RandomRotation(180),
             transforms.RandomHorizontalFlip(0.5),  
             transforms.RandomVerticalFlip(0.5),  
             ]
        )
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        self.basic_transform = transforms.Compose(
            [transforms.Resize((224, 224), antialias=True),  # resize to 256x256 square
             transforms.ConvertImageDtype(torch.float),
             transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD) 
             ]
        )
        print('Loading img...')
        if mode != 'train':
            self.wsi_ids = [self.wsi_ids_lst[fold]]
            self.test_sample = self.wsi_ids[0]

        else:
            self.wsi_ids = list(set(self.wsi_ids_lst) - set([self.wsi_ids_lst[fold]]))  #31
        if data_name=='her2':
            self.patch_meta = {wsi_i:[j for j in os.listdir(self.patch_path) if j.split('_')[0] == wsi_i] for wsi_i in self.wsi_ids}
            self.patch_ids = []
            for wsi_i in list(self.patch_meta.keys()):
                self.patch_ids.extend(self.patch_meta[wsi_i])
            print('Loading orig exp...')
            self.exps = {patch_i: torch.Tensor(self.get_exp(patch_i.split('_')[0])[patch_i.split('.')[0]]) for patch_i
                         in
                         self.patch_ids}
        elif data_name=='cscc':  
            self.patch_meta = {wsi_i: [j for j in os.listdir(self.patch_path) if '_'.join(j.split('_')[:-1]) == wsi_i] for wsi_i in
                               self.wsi_ids}
            self.patch_ids = []
            for wsi_i in list(self.patch_meta.keys()):
                self.patch_ids.extend(self.patch_meta[wsi_i])
            print('Loading orig exp...')
            self.exps = {patch_i: torch.Tensor(self.get_exp('_'.join(patch_i.split('_')[:-1]))[patch_i.split('.')[0]]) for patch_i in
                         self.patch_ids}
    def __getitem__(self, item):
 
        patch_name = list(self.exps.keys())[item]
        patch = self.get_img(patch_name)  #(3,224,224)
        exp = self.exps[patch_name]
        patch_orig_path = os.path.join(self.orig_path,patch_name.replace('png','npy'))
        orig_exp = torch.Tensor(np.load(patch_orig_path)).squeeze()
        data = (patch_name,patch,orig_exp,exp)
        return data


    def __len__(self):
        return len(self.patch_ids)

    def get_img(self,img_name):
        img_path = os.path.join(self.patch_path,img_name)
        img = Image.open(img_path)
        img = torch.Tensor(np.array(img)/255)
        img = img.permute(1, 0, 2)  
        img = img.permute(2,0,1)
        return self.transforms(img)
    def get_exp(self,wsi_id):
        label_path = os.path.join(self.label_path,wsi_id+'.json')
        with open(label_path,'r') as f:
            exp_file = json.load(f)
        return exp_file

    def transforms(self,img):
        my_transforms = [self.basic_transform, self.train_transform]
        if self.mode!='train':
            return my_transforms[0](img)
        else:
            return my_transforms[1](my_transforms[0](img))

class Initiate_dataset_with_center(Dataset):
    def __init__(self,patch_path,exp_label_path,orig_path,center_path,mode,fold,data_name):
        super(Initiate_dataset_with_center,self).__init__()
        self.patch_path = patch_path
        self.label_path = exp_label_path
        self.orig_path = orig_path
        self.center_path = center_path
        self.mode = mode
        self.wsi_ids_lst = [i.split('.')[0] for i in os.listdir(exp_label_path)]
        self.wsi_ids_lst.sort()
        self.train_transform = transforms.Compose(
            [transforms.RandomRotation(180),
             transforms.RandomHorizontalFlip(0.5),  
             transforms.RandomVerticalFlip(0.5), 
             ]
        )
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        self.basic_transform = transforms.Compose(
            [transforms.Resize((224, 224), antialias=True), 
             transforms.ConvertImageDtype(torch.float),
             transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)  
             ]
        )
        print('Loading img...')
        if mode != 'train':
            self.wsi_ids = [self.wsi_ids_lst[fold]]
            self.test_sample = self.wsi_ids[0]
            print(self.test_sample)

        else:
            self.wsi_ids = list(set(self.wsi_ids_lst) - set([self.wsi_ids_lst[fold]])) 
        if data_name=='her2':
            self.patch_meta = {wsi_i:[j for j in os.listdir(self.patch_path) if j.split('_')[0] == wsi_i] for wsi_i in self.wsi_ids}
            self.patch_ids = []
            for wsi_i in list(self.patch_meta.keys()):
                self.patch_ids.extend(self.patch_meta[wsi_i])
            print('Loading orig exp...')
            self.exps = {patch_i: torch.Tensor(self.get_exp(patch_i.split('_')[0])[patch_i.split('.')[0]]) for patch_i in self.patch_ids}
            self.centers = {patch_i:torch.Tensor(self.get_center(patch_i.split('_')[0])[patch_i.split('.')[0]]) for patch_i in self.patch_ids}
        elif data_name=='cscc':
            self.patch_meta = {wsi_i: [j for j in os.listdir(self.patch_path) if '_'.join(j.split('_')[:-1]) == wsi_i]
                               for wsi_i in
                               self.wsi_ids}
            self.patch_ids = []
            for wsi_i in list(self.patch_meta.keys()):
                self.patch_ids.extend(self.patch_meta[wsi_i])
            print('Loading orig exp...')
            self.exps = {patch_i: torch.Tensor(self.get_exp('_'.join(patch_i.split('_')[:-1]))[patch_i.split('.')[0]])
                         for patch_i in
                         self.patch_ids}
            self.centers = {patch_i: torch.Tensor(self.get_center('_'.join(patch_i.split('_')[:-1]))[patch_i.split('.')[0]]) for
                            patch_i in self.patch_ids}

    def __getitem__(self, item):

        patch_name = list(self.exps.keys())[item]
        patch = self.get_img(patch_name)  #(3,224,224)
        exp = self.exps[patch_name]
        patch_orig_path = os.path.join(self.orig_path,patch_name.replace('png','npy'))
        orig_exp = torch.Tensor(np.load(patch_orig_path)).squeeze()
        center = self.centers[patch_name]
        data = (patch_name,patch,orig_exp,exp,center)
        return data


    def __len__(self):
        return len(self.patch_ids)

    def get_img(self,img_name):
        img_path = os.path.join(self.patch_path,img_name)
        img = Image.open(img_path)
        img = torch.Tensor(np.array(img)/255)
        img = img.permute(1, 0, 2) 
        img = img.permute(2,0,1)
        return self.transforms(img)
    def get_exp(self,wsi_id):
        label_path = os.path.join(self.label_path,wsi_id+'.json')
        with open(label_path,'r') as f:
            exp_file = json.load(f)
        return exp_file
    def get_center(self,wsi_id):
        center_path = os.path.join(self.center_path,wsi_id+'.json')
        with open(center_path,'r') as f:
            center_file = json.load(f)
        return center_file

    def transforms(self,img):
        my_transforms = [self.basic_transform, self.train_transform]
        if self.mode!='train':
            return my_transforms[0](img)
        else:
            return my_transforms[1](my_transforms[0](img))



#直接定义一个api，来返回dataloader
def load_dataset(config): 
    patch_path = config['patch_path']
    exp_label_path = config['exp_label_path']
    orig_path = config['orig_path']
    test_sample = config['test_index']

    train_dataset = Initiate_dataset(patch_path,exp_label_path,orig_path,'train',test_sample,config['data_name'])
    test_dataset = Initiate_dataset(patch_path,exp_label_path,orig_path,'test',test_sample,config['data_name'])
    train_loader = DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True,drop_last=True,num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset,batch_size=config['batch_size'],shuffle=False,num_workers=5,drop_last=True)
    test_sample = test_dataset.test_sample
    return train_loader,test_loader,test_sample

if __name__ == '__main__':
    patch_path = './01-data/01-her2数据集/01-gen-patch/'
    exp_label_path = './01-data/01-her2数据集/02-gene-exp-label/'
    ref_path = './01-data/01-her2数据集/04-singlecell_ref/filter_her2_ref.csv'
    predicted_gene_path = './01-HER2+/her2st-master/data/her_hvg_cut_1000.npy'
    seg_path = './01-data/01-her2数据集/03-patch-segResult/json/'
    orig_path = '/./01-data/01-her2数据集/05-orig_exp/'
    data_test = HER2_dataset(patch_path,exp_label_path,predicted_gene_path,ref_path,orig_path,seg_path,'test',0)
    for item in data_test:
        print(item)






