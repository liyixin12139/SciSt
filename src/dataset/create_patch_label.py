import json
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
from PIL import ImageFile, Image
import torch
import os
import numpy as np
from collections import defaultdict as dfd
import scprep as scp
import pandas as pd
from torch.utils.data import DataLoader
from paperTCGN.TCGN_model.graph_construction import calcADJ
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, fold=0, r=2, flatten=True, ori=False, adj=False, prune='Grid', neighs=4): #后两个参数用来构建邻接矩阵
        super(ViT_HER2ST, self).__init__()
        #r指的是每个spot长和宽的一半，也就是半径
        self.cnt_dir = './data/ST-cnts'  #基因表达数据
        self.img_dir = './data/ST-imgs' #图像数据
        self.pos_dir = './data/ST-spotfiles'#spot对应在图像上的坐标文件
        self.r = 224 // r  

        gene_list = np.load(
            './hvg_genes.npy',
            allow_pickle=True).tolist()
        self.gene_list = gene_list #len(gene_list) = 785
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]  #列出所有section的名字，例如A1，B1
        self.train = train
        self.ori = ori
        self.adj = adj

        samples = names[1:33]  #这里已经删除了4例不符合的section，所以只有32张    sample对应的是section 剔除了A1，所以从1:33
        te_names = [samples[fold]]  #用来测试的sample的name，用list包起来是为了方便后续数据格式的处理，都用一样的
        print(te_names)
        tr_names = list(set(samples) - set(te_names))  #剩下的section用来训练


        if train:
            self.names = tr_names
        else:
            self.names = te_names   #后续只用self.names指代用来处理的数据，无论是train还是test
        self.names = samples

        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in self.names}  #{'A2':torch.tensor([...])} 打开的是全sample的图像对应的数字
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names} #meta是基因表达及在图像上对应坐标位置的pandas数据

        self.gene_set = list(gene_list)
        self.exp_dict = {
            i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))  #从所有表达基因中选出要的785个基因,并进行归一化处理
            for i, m in self.meta_dict.items()
        }   #exp_dict每个value的形状（325，785），325指spot个数

        self.center_dict = {
            i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)  #np.floor用来把坐标的小数点去掉,并转成int形式
            for i, m in self.meta_dict.items()
        }
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}  #x和y指的是x轴和y轴上的第几个spot
        self.adj_dict = {
            i: calcADJ(m, neighs, pruneTag=prune)
            for i, m in self.loc_dict.items()
        }   #这里涉及到邻接矩阵的构建，m指的是每一个sample的mata数据    这部分稍后看(简单来说就是某两个spot之间的距离小于某一值的话就变成1，返回的是一个仅有0和1的torch张量)
        self.patch_dict = dfd(lambda: None)
        self.lengths = [len(i) for i in self.meta_dict.values()]  #每一个sample的spot个数，也就是长度
        self.cumlen = np.cumsum(self.lengths)  #计算累加和，返回是一个矩阵，是前面所有spot数与当前spot数的和
        self.id2name = dict(enumerate(self.names))
        self.flatten = flatten

        self.id_dict = {
            i: list(m.index)  #从所有表达基因中选出要的785个基因,并进行归一化处理
            for i, m in self.meta_dict.items()
        }



    def __getitem__(self, index):
        ID = self.id2name[index]
        im = self.img_dict[ID] #从img字典里加载大图像矩阵
        im = im.permute(1, 0, 2)   #是为了对应x和y坐标切patch
        exps = self.exp_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]  #现在还是None
        positions = torch.LongTensor(loc)
        patch_dim = 3 * 2* self.r *2* self.r  #得到每张图像如果flatten有多少维。 3是通道数
        _id=self.id_dict[ID]
        exps = torch.Tensor(exps)
        patch_name = list()
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches, patch_dim)) #flatten就是所有patch不按照图像来处理，而是拉成一维向量来处理
            else:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r)) 

            for i in range(n_patches):
                center = centers[i]  #得到每张patch中心坐标位置
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :] #从中心位置延伸半径切成224 X 224
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i] = patch.permute(2, 0, 1)  #把通道数放在最前面
            self.patch_dict[ID] = patches
        data = [patches, positions, exps,_id]  #position用的是x和y，而不是pixel    exps用的是基因表达量归一化病log后的值
        if self.adj:
            data.append(adj)
        data.append(torch.Tensor(centers))
        data.append(ID)
        return data   #data返回的是一个sample的所有数据，包括图像、位置和基因表达

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]   
        path = pre + '/' + fig_name
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.tsv'
        df = pd.read_csv(path, sep='\t', index_col=0)

        return df

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))  #setindex指的是通过指定的列，即‘id’来连接两个df，此时返回的meta文件中，cnt和坐标是连在一起的
        return meta


class CSCC_ST(torch.utils.data.Dataset):
    def __init__(self, train=True, fold=0, r=2, flatten=True, ori=False, adj=False, prune='Grid', neighs=4): #后两个参数用来构建邻接矩阵
        super(CSCC_ST, self).__init__()
        self.r = 224 // r
        scc_data_path = './SCC_data/'
        patients = ['P2', 'P5', 'P9', 'P10'] #构造cscc数据集的sample列表
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i + '_ST_' + j)

        self.cnt_dir = os.path.join(scc_data_path,'cnt_dir') #基因表达数据
        self.img_dir = os.path.join(scc_data_path,'img_dir')#图像数据
        self.pos_dir =os.path.join(scc_data_path,'pos_dir')

        gene_list = list(
            np.load(os.path.join(scc_data_path,'skin_hvg_cut_1000.npy'),
                    allow_pickle=True))  
        self.gene_list = gene_list  # len(gene_list) = 169

        self.train = train
        self.ori = ori
        self.adj = adj
        # samples = ['P2_ST_rep1','P2_ST_rep2','P2_ST_rep3','P5_ST_rep1']
        samples = names

        te_names = [samples[fold]]  # 用来测试的sample的name，用list包起来是为了方便后续数据格式的处理，都用一样的
        print('test_sample: ',te_names)
        tr_names = list(set(samples) - set(te_names))  # 剩下的section用来训练

        if train:
            self.names = tr_names
        else:
            self.names = te_names
        self.names = samples


        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in
                         self.names}  # {'A2':torch.tensor([...])} 打开的是全sample的图像对应的数字

        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names}  # meta是基因表达及在图像上对应坐标位置的pandas数据


        self.gene_set = list(gene_list)
        self.exp_dict = {
            i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
            # 从所有表达基因中选出要的785个基因,并进行归一化处理
            for i, m in self.meta_dict.items()
        }  # exp_dict每个value的形状（325，785），325指spot个数
        self.center_dict = {
            i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)  # np.floor用来把坐标的小数点去掉,并转成int形式
            for i, m in self.meta_dict.items()
        }
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}  # x和y指的是x轴和y轴上的第几个spot
        self.adj_dict = {
            i: calcADJ(m, neighs, pruneTag=prune)
            for i, m in self.loc_dict.items()
        }  # 这里涉及到邻接矩阵的构建，m指的是每一个sample的mata数据    这部分稍后看(简单来说就是某两个spot之间的距离小于某一值的话就变成1，返回的是一个仅有0和1的torch张量)
        self.patch_dict = dfd(lambda: None)
        self.lengths = [len(i) for i in self.meta_dict.values()]  # 每一个sample的spot个数，也就是长度
        self.cumlen = np.cumsum(self.lengths)  # 计算累加和，返回是一个矩阵，是前面所有spot数与当前spot数的和
        self.id2name = dict(enumerate(self.names))
        self.flatten = flatten

    def __getitem__(self, index):
        ID = self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1, 0, 2)  # 为了对应x和y坐标切patch
        exps = self.exp_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * 2 * self.r * 2 * self.r
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches, patch_dim))  # flatten就是所有patch不按照图像来处理，而是拉成一维向量来处理
            else:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))

            for i in range(n_patches):
                center = centers[i]  # center指的是中心坐标的位置
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i] = patch.permute(2, 0, 1)  # 把通道数放在最前面
            self.patch_dict[ID] = patches
        data = [patches, positions, exps]  # position用的是x和y，而不是pixel    exps用的是基因表达量归一化病log后的值
        if self.adj:
            data.append(adj)
        if self.ori:
            data += [torch.Tensor(oris), torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        data.append(ID)
        return data  # data返回的是一个sample的所有数据，包括图像、位置和基因表达

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):#->(15872,15872,3)
        pre = self.img_dir + '/' + name+ '.jpg'
        im = Image.open(pre)
        return im

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '_stdata.tsv'
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')),how='inner')# setindex指的是通过指定的列，即‘id’来连接两个df，此时返回的meta文件中，cnt和坐标是连在一起的
        return meta


def pk_load(fold,mode='train',flatten=False,dataset='her2st',r=2,ori=False,adj=False,prune='Grid',neighs=4):
    '''
    定义一个api，直接返回dataset
    '''
    assert dataset in ['her2st','cscc']
    if dataset=='her2st':
        dataset = ViT_HER2ST(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    elif dataset=='cscc':  
        dataset = CSCC_ST(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    return dataset

def HER_CSCC_gen_origPatch_Label_Center(patch_save_path,label_save_path,center_save_path,dataset_name):
    trainset = pk_load(0, 'train', False, dataset_name, neighs=4, prune='Grid') #从api返回dataset
    train_loader = DataLoader(trainset, batch_size=1, num_workers=10, shuffle=False)   #batch size只能是1
    for item in train_loader: 
        _id = item[-1][0]
        imgs = item[0].squeeze(0)  #(n,3,224,224)
        exps = item[1].squeeze(0) #(n,785)
        centers = item[-2].squeeze(0)
        centers_dict = {}
        exp_file = {}
        for i_img in range(imgs.shape[0]): #从dataloader依次返回patch级数据，保存
            save_name = _id+'_'+str(i_img)
            centers_dict[save_name] = np.array(centers[i_img]).tolist()
            img = np.array(imgs[i_img].permute(2,1,0)) #转为符合cv2保存的顺序
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #修改颜色通道，否则可视化有问题
            save_name = _id+'_'+str(i_img)
            cv2.imwrite(os.path.join(patch_save_path,save_name+'.png'),img)
            exp_file[save_name] = np.array(exps[i_img]).tolist()
        with open(os.path.join(center_save_path,_id+'.json'),'w') as f:
            json.dump(centers_dict,f)
        with open(os.path.join(label_save_path,_id+'.json'),'w') as f:
            json.dump(exp_file,f)

if __name__=="__main__":
    patch_save_path='./01_patch'
    label_save_path='./02_label'
    center_save_path='./03_center'
    HER_CSCC_gen_origPatch_Label_Center(patch_save_path,label_save_path,center_save_path,'herst') #['her2st','cscc']