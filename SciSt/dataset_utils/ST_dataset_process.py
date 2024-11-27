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

    def __init__(self, train=True, fold=0, r=4, flatten=True, ori=False, adj=False, prune='Grid', neighs=4):
        super(ViT_HER2ST, self).__init__()
        #r指的是每个spot长和宽的一半，也就是半径
        self.cnt_dir = './01-HER2+/her2st-master/data/ST-cnts'  #基因表达数据
        self.img_dir = './01-HER2+/her2st-master/data/ST-imgs' #图像数据
        self.pos_dir = './01-HER2+/her2st-master/data/ST-spotfiles'#spot对应在图像上的坐标文件
        self.lbl_dir = './01-HER2+/her2st-master/data/ST-pat/lbl'#有patch对应annotation的文件
        self.r = 224 // r    

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('/2data/liyixin/HE2ST/02dataset/公共数据集/01-HER2+/her2st-master/data/her_hvg_cut_1000.npy', allow_pickle=True))    #这个文件是怎么生成的？这个文件是选出来的785个基因list，我是不是可以直接用？
        self.gene_list = gene_list 
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]  
        self.train = train
        self.ori = ori
        self.adj = adj
       
        samples = names[1:33]  
        te_names = [samples[fold]]  
        print(te_names)
        tr_names = list(set(samples) - set(te_names))  


        if train:
            self.names = tr_names
            # self.names = samples
        else:
            self.names = te_names  
        self.names = samples

        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in self.names}  
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names} 
        self.label = {i: None for i in self.names} 
        self.lbl2id = {   
            'invasive cancer': 0, 'breast glands': 1, 'immune infiltrate': 2,
            'cancer in situ': 3, 'connective tissue': 4, 'adipose tissue': 5, 'undetermined': -1
        }
        if not train and self.names[0] in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']: 
            self.lbl_dict = {i: self.get_lbl(i) for i in self.names}

            idx = self.meta_dict[self.names[0]].index   
            lbl = self.lbl_dict[self.names[0]]
            lbl = lbl.loc[idx, :]['label'].values 
            self.label[self.names[0]] = lbl 
        elif train:
            for i in self.names:
                idx = self.meta_dict[i].index
                if i in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
                    lbl = self.get_lbl(i)
                    lbl = lbl.loc[idx, :]['label'].values
                    lbl = torch.Tensor(list(map(lambda i: self.lbl2id[i], lbl)))
                    self.label[i] = lbl  
                else:
                    self.label[i] = torch.full((len(idx),), -1) 
        self.gene_set = list(gene_list)
        self.exp_dict = {
            i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))  
            for i, m in self.meta_dict.items()
        }  

        if self.ori:
            self.ori_dict = {i: m[self.gene_set].values for i, m in self.meta_dict.items()}  
            self.counts_dict = {}
            for i, m in self.ori_dict.items():
                n_counts = m.sum(1)  
                sf = n_counts / np.median(n_counts)  
                self.counts_dict[i] = sf  
        self.center_dict = {
            i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)  
            for i, m in self.meta_dict.items()
        }
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}  
        self.adj_dict = {
            i: calcADJ(m, neighs, pruneTag=prune)
            for i, m in self.loc_dict.items()
        }  
        self.patch_dict = dfd(lambda: None)
        self.lengths = [len(i) for i in self.meta_dict.values()] 
        self.cumlen = np.cumsum(self.lengths)  
        self.id2name = dict(enumerate(self.names))
        self.flatten = flatten

        self.id_dict = {
            i: list(m.index)  
            for i, m in self.meta_dict.items()
        }



    def __getitem__(self, index):
        ID = self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1, 0, 2)   

        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]  
            sfs = self.counts_dict[ID] 

        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        label = self.label[ID]
        _id=self.id_dict[ID]
        exps = torch.Tensor(exps)
        patch_name = list()
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches, patch_dim)) 
            else:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))

            for i in range(n_patches):
                center = centers[i]  
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i] = patch.permute(2, 0, 1)  
            self.patch_dict[ID] = patches
        data = [patches, positions, exps,_id]  
        if self.adj:
            data.append(adj)
        if self.ori:
            data += [torch.Tensor(oris), torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        data.append(ID)
        return data   

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
        meta = cnt.join((pos.set_index('id')))  
        return meta

    def get_lbl(self, name):
        path = self.lbl_dir + '/' + name + '_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id', inplace=True)
        return df

class CSCC_ST(torch.utils.data.Dataset):
    def __init__(self, train=True, fold=0, r=4, flatten=True, ori=False, adj=False, prune='Grid', neighs=4): 
        super(CSCC_ST, self).__init__()
        self.r = 224 // r
        scc_data_path = '/2data/liyixin/HE2ST/02dataset/公共数据集/9-鳞状细胞癌-GSE144239_RAW-但分辨率较低-可能做测试用/SCC_data/'
        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i + '_ST_' + j)

        self.cnt_dir = os.path.join(scc_data_path,'cnt_dir') 
        self.img_dir = os.path.join(scc_data_path,'img_dir')
        self.pos_dir =os.path.join(scc_data_path,'pos_dir')

        gene_list = list(
            np.load(os.path.join(scc_data_path,'skin_hvg_cut_1000.npy'),
                    allow_pickle=True))  
        self.gene_list = gene_list 
        self.gene_list.remove('AC013461.1')
        self.gene_list.remove('NEFL')


        self.train = train
        self.ori = ori
        self.adj = adj
        # samples = ['P2_ST_rep1','P2_ST_rep2','P2_ST_rep3','P5_ST_rep1']
        samples = names

        te_names = [samples[fold]]  
        print('test_sample: ',te_names)
        tr_names = list(set(samples) - set(te_names))  

        if train:
            self.names = tr_names
            # self.names = samples
        else:
            self.names = te_names
        self.names = samples


        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in
                         self.names}  

        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names}  


        self.gene_set = list(gene_list)
        self.exp_dict = {
            i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
            for i, m in self.meta_dict.items()
        }  

        if self.ori:
            self.ori_dict = {i: m[self.gene_set].values for i, m in self.meta_dict.items()} 
            self.counts_dict = {}
            for i, m in self.ori_dict.items():
                n_counts = m.sum(1)  
                sf = n_counts / np.median(n_counts) 
                self.counts_dict[i] = sf  
        self.center_dict = {
            i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)  
            for i, m in self.meta_dict.items()
        }
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}  
        self.adj_dict = {
            i: calcADJ(m, neighs, pruneTag=prune)
            for i, m in self.loc_dict.items()
        }  
        self.patch_dict = dfd(lambda: None)
        self.lengths = [len(i) for i in self.meta_dict.values()]  
        self.cumlen = np.cumsum(self.lengths) 
        self.id2name = dict(enumerate(self.names))
        self.flatten = flatten

    def __getitem__(self, index):
        ID = self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1, 0, 2)  

        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]  
            sfs = self.counts_dict[ID]  

        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches, patch_dim))  
            else:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))

            for i in range(n_patches):
                center = centers[i]  
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i] = patch.permute(2, 0, 1) 
            self.patch_dict[ID] = patches
        data = [patches, positions, exps] 
        if self.adj:
            data.append(adj)
        if self.ori:
            data += [torch.Tensor(oris), torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        data.append(ID)
        return data  

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
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
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
        meta = cnt.join((pos.set_index('id')),how='inner')
        return meta


class Breast_ST(torch.utils.data.Dataset):
    def __init__(self, train=True, fold=0, r=2):
        super(Breast_ST, self).__init__()
        self.r = 224 // r
        breast_data_path = './5-STNet-有点糊-Human breast cancer in situ capturing transcriptomics/'
        names = list(np.load(os.path.join(breast_data_path,'sample_names.npy'),allow_pickle=True))
        names.sort()
        self.cnt_dir = os.path.join(breast_data_path,'cnt_dir_orig') #基因表达数据
        self.img_dir = os.path.join(breast_data_path,'img_dir')#图像数据
        self.pos_dir =os.path.join(breast_data_path,'pos_dir') #中心坐标数据

        gene_list = list(
            np.load('./5-STNet-有点糊-Human breast cancer in situ capturing transcriptomics/predicted_genes/breast_hvg_cut_1000.npy',
                    allow_pickle=True)) 
        self.gene_list = gene_list  

        self.train = train

        samples=names
        te_names = [samples[fold]] 
        print('test_sample: ',te_names)
        tr_names = list(set(samples) - set(te_names)) 

        if train:
            self.names = tr_names
            # self.names = samples
        else:
            self.names = te_names
        self.names = samples


        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in
                         self.names}  

        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names}  
        self.gene_set = list(gene_list)

        self.exp_dict = {
            i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
            for i, m in self.meta_dict.items()
        }  


        self.center_dict = {
            i: np.floor(m[['X', 'Y']].values).astype(int)  
            for i, m in self.meta_dict.items()
        }


        self.patch_dict = dfd(lambda: None)
        self.lengths = [len(i) for i in self.meta_dict.values()] 
        self.cumlen = np.cumsum(self.lengths)  
        self.id2name = dict(enumerate(self.names))


    def __getitem__(self, index):
        ID = self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1, 0, 2) 
        exps = self.exp_dict[ID]
        centers = self.center_dict[ID]
        patches = self.patch_dict[ID]
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))

            for i in range(n_patches):
                center = centers[i]  
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.permute(2, 0, 1)  
            self.patch_dict[ID] = patches
        data = [patches, exps]  
        data.append(torch.Tensor(centers))
        data.append(ID)
        return data  

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):#->(15872,15872,3)
        pre = self.img_dir + '/HE_' + name+ '.jpg'
        im = Image.open(pre)
        return im

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '_stdata.tsv'
        if not os.path.exists(path):
            name = name.replace('BT','BC')
            path = self.cnt_dir + '/' + name + '_stdata.tsv'
        df = pd.read_csv(path,sep='\t')
        df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
        return df

    def get_pos(self, name):
        path = self.pos_dir + '/spots_' + name + '.csv'
        df = pd.read_csv(path)
        df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
        return df

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = pd.merge(cnt, pos, on='id', how='right')
        meta = meta.set_index('id',drop=True)
        meta.dropna(axis=0,how='any',inplace=True)
        return meta



def pk_load(fold,mode='train',flatten=False,dataset='her2st',r=2,ori=False,adj=False,prune='Grid',neighs=4):
    assert dataset in ['her2st','cscc','breast']
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
    elif dataset == 'breast':  
        dataset = Breast_ST(
            train=(mode == 'train'), fold=fold, r=r
        )

    return dataset

def HER_CSCC_gen_origPatch_Label_Center():

    trainset = pk_load(0, 'test', False, 'cscc', neighs=4, prune='Grid')
    train_loader = DataLoader(trainset, batch_size=1, num_workers=3, shuffle=False)  
    save_path = './02-单细胞参考预测基因表达/01-data/03-CSCC/01-gen-patch'
    gene_save_path = './02-单细胞参考预测基因表达/01-data/03-CSCC/02-gene-exp-label/'
    center_save_path = './02-单细胞参考预测基因表达/01-data/03-CSCC/06-center/'
    for item in train_loader:
        _id = item[-1][0]
        imgs = item[0].squeeze(0)  #(n,3,224,224)
        exps = item[1].squeeze(0) #(n,785)
        centers = item[-2].squeeze(0)
        centers_dict = {}
        exp_file = {}
        for i_img in range(imgs.shape[0]):
            save_name = _id+'_'+str(i_img)
            centers_dict[save_name] = np.array(centers[i_img]).tolist()
            img = np.array(imgs[i_img].permute(2,1,0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            save_name = _id+'_'+str(i_img)
            cv2.imwrite(os.path.join(save_path,save_name+'.png'),img)
            exp_file[save_name] = np.array(exps[i_img]).tolist()
        with open(os.path.join(center_save_path,_id+'.json'),'w') as f:
            json.dump(centers_dict,f)
        with open(os.path.join(gene_save_path,_id+'.json'),'w') as f:
            json.dump(exp_file,f)




def compute_tls_score(prop: pd.DataFrame,
                      b_name: str = "B-cells",
                      t_name: str = "T-cells",
                      ) -> np.ndarray:
    """compute tls score from prop file"""

    n_spots = prop.shape[0]
    pos_1 = np.argmax(prop.columns == b_name)  
    pos_2 = np.argmax(prop.columns == t_name)

    jprod = np.zeros(n_spots)

    print("computing TLS-Score")
    for s in range(n_spots):
        vec = prop.values[s, :].reshape(-1, 1)
        prod = np.dot(vec, vec.T)
        nprod = prod / prod.sum()
        N = prod.shape[0]  
        jprod[s] = nprod[pos_1, pos_2] * 2  
        jprod[s] -= (nprod.sum() / (0.5 * (N ** 2 + N)))

    jprod = pd.DataFrame(jprod,
                         index=prop.index,
                         columns=['probability'])

    return jprod

def patch_spot_match_dict():
    trainset = pk_load(0, 'test', False, 'her2st', neighs=4, prune='Grid')
    train_loader = DataLoader(trainset, batch_size=1, num_workers=3, shuffle=False)  # batch size只能是1
    patch_spot_dict={}
    for item in train_loader:
        _id = item[-1][0]
        imgs = item[0].squeeze(0)  #(n,3,224,224)
        ids=item[3]
        for i_img in range(imgs.shape[0]):
            save_name = _id+'_'+str(i_img)
            patch_spot_dict[str(save_name)]=ids[i_img][0]
    with open('./02-单细胞参考预测基因表达/01-data/01-her2数据集/patch_spot_match.json','w') as f:
        json.dump(patch_spot_dict,f)



if __name__=="__main__":
    patch_spot_match_dict()

















