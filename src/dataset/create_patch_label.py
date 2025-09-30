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
def calcADJ(coord, k=8, distanceType='euclidean', pruneTag='NA'):
    #coord的形状是（num_spots,2）
    #计算的是欧几里得距离，也就是spot之间的绝对距离
    r"""
    Calculate spatial Matrix directly use X/Y coordinates
    """
    spatialMatrix=coord#.cpu().numpy()
    nodes=spatialMatrix.shape[0]
    Adj=torch.zeros((nodes,nodes))
    for i in np.arange(spatialMatrix.shape[0]):
        tmp=spatialMatrix[i,:].reshape(1,-1)
        distMat = distance.cdist(tmp,spatialMatrix, distanceType)
        if k == 0:
            k = spatialMatrix.shape[0]-1
        res = distMat.argsort()[:k+1]
        tmpdist = distMat[0,res[0][1:k+1]]
        boundary = np.mean(tmpdist)+np.std(tmpdist) #optional
        for j in np.arange(1,k+1):
            # No prune
            if pruneTag == 'NA':
                Adj[i][res[0][j]]=1.0
            elif pruneTag == 'STD':
                if distMat[0,res[0][j]]<=boundary:
                    Adj[i][res[0][j]]=1.0
            # Prune: only use nearest neighbor as exact grid: 6 in cityblock, 8 in euclidean
            elif pruneTag == 'Grid':
                if distMat[0,res[0][j]]<=2.0:
                    Adj[i][res[0][j]]=1.0
    return Adj

class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, fold=0, r=2, flatten=True, ori=False, adj=False, prune='Grid', neighs=4): # The last two parameters are used to construct the adjacency matrix
        super(ViT_HER2ST, self).__init__()
        # `r` is half the spot width/height (i.e., the radius)
        self.cnt_dir = './data/ST-cnts'  # gene expression data
        self.img_dir = './data/ST-imgs' # image data
        self.pos_dir = './data/ST-spotfiles'# file of spot coordinates on the image
        self.r = 224 // r  

        gene_list = np.load(
            './hvg_genes.npy',
            allow_pickle=True).tolist()
        self.gene_list = gene_list #len(gene_list) = 785
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]  # List all section names, e.g., A1, B1
        self.train = train
        self.ori = ori
        self.adj = adj

        samples = names[1:33]  # Four invalid sections have been removed, so there are 32 slides. `sample` corresponds to section; A1 is excluded, so use indices 1:33
        te_names = [samples[fold]]  # Name of the sample for testing; wrap it in a list to standardize downstream data handling
        print(te_names)
        tr_names = list(set(samples) - set(te_names))  # The remaining sections are used for training


        if train:
            self.names = tr_names
        else:
            self.names = te_names   # Use `self.names` to refer to the data to process, whether train or test
        self.names = samples

        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in self.names}  # e.g., {'A2': torch.tensor([...])}; loaded image arrays for all samples
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names} # `meta` is a pandas DataFrame of gene expression and the corresponding image coordinates

        self.gene_set = list(gene_list)
        self.exp_dict = {
            i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))  # Select the 785 target genes from all expressed genes and normalize
            for i, m in self.meta_dict.items()
        }   # Each `exp_dict` value has shape (325, 785); 325 is the number of spots

        self.center_dict = {
            i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)  # `np.floor` strips decimals from coordinates and casts to int
            for i, m in self.meta_dict.items()
        }
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}  # `x` and `y` denote the spot index along the x- and y-axes
        self.adj_dict = {
            i: calcADJ(m, neighs, pruneTag=prune)
            for i, m in self.loc_dict.items()
        }   # Adjacency matrix construction: `m` is the meta data for each sample. In short, two spots closer than a threshold are connected (1); returns a binary torch tensor
        self.patch_dict = dfd(lambda: None)
        self.lengths = [len(i) for i in self.meta_dict.values()]  # Number of spots per sample (its length)
        self.cumlen = np.cumsum(self.lengths)  # Cumulative sum: a vector giving the sum of prior spot counts and the current one
        self.id2name = dict(enumerate(self.names))
        self.flatten = flatten

        self.id_dict = {
            i: list(m.index)  # Select the 785 target genes from all expressed genes and normalize
            for i, m in self.meta_dict.items()
        }



    def __getitem__(self, index):
        ID = self.id2name[index]
        im = self.img_dict[ID] # Load the large image array from the `img` dictionary
        im = im.permute(1, 0, 2)   # Permute to slice patches along x and y coordinates
        exps = self.exp_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]  # Currently None
        positions = torch.LongTensor(loc)
        patch_dim = 3 * 2* self.r *2* self.r  # Compute the flattened dimension per image; 3 is the channel count
        _id=self.id_dict[ID]
        exps = torch.Tensor(exps)
        patch_name = list()
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches, patch_dim)) # `flatten` means treating all patches as 1D vectors rather than image grids
            else:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r)) 

            for i in range(n_patches):
                center = centers[i]  # Get the center coordinate for each patch
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :] # Crop 224×224 patches centered at the spot using the radius
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i] = patch.permute(2, 0, 1)  # Move the channel dimension to the front
            self.patch_dict[ID] = patches
        data = [patches, positions, exps,_id]  # `position` uses (x, y) indices (not pixels); `exps` uses normalized and log-transformed gene expression
        if self.adj:
            data.append(adj)
        data.append(torch.Tensor(centers))
        data.append(ID)
        return data   # `data` for one sample: images, positions, and gene expressions

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
        meta = cnt.join((pos.set_index('id')))  # `set_index` joins by the 'id' column; the resulting `meta` contains both counts (`cnt`) and coordinates
        return meta


class CSCC_ST(torch.utils.data.Dataset):
    def __init__(self, train=True, fold=0, r=2, flatten=True, ori=False, adj=False, prune='Grid', neighs=4): # The last two parameters are used to construct the adjacency matrix
        super(CSCC_ST, self).__init__()
        self.r = 224 // r
        scc_data_path = './SCC_data/'
        patients = ['P2', 'P5', 'P9', 'P10'] # Build the sample list for the CSCC dataset
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i + '_ST_' + j)

        self.cnt_dir = os.path.join(scc_data_path,'cnt_dir') # gene expression data
        self.img_dir = os.path.join(scc_data_path,'img_dir')# image data
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

        te_names = [samples[fold]]  # Name of the sample for testing; wrap it in a list to standardize downstream data handling
        print('test_sample: ',te_names)
        tr_names = list(set(samples) - set(te_names))  # The remaining sections are used for training

        if train:
            self.names = tr_names
        else:
            self.names = te_names
        self.names = samples


        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in
                         self.names}  # e.g., {'A2': torch.tensor([...])}; loaded image arrays for all samples

        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names}  # `meta` is a pandas DataFrame of gene expression and the corresponding image coordinates


        self.gene_set = list(gene_list)
        self.exp_dict = {
            i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
            # Select the 785 target genes from all expressed genes and normalize
            for i, m in self.meta_dict.items()
        }  # Each `exp_dict` value has shape (325, 785); 325 is the number of spots
        self.center_dict = {
            i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)  # `np.floor` strips decimals from coordinates and casts to int
            for i, m in self.meta_dict.items()
        }
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}  # `x` and `y` denote the spot index along the x- and y-axes
        self.adj_dict = {
            i: calcADJ(m, neighs, pruneTag=prune)
            for i, m in self.loc_dict.items()
        }  # Adjacency matrix construction: `m` is the meta data for each sample. In short, two spots closer than a threshold are connected (1); returns a binary torch tensor
        self.patch_dict = dfd(lambda: None)
        self.lengths = [len(i) for i in self.meta_dict.values()]  # Number of spots per sample (its length)
        self.cumlen = np.cumsum(self.lengths)  # Cumulative sum: a vector giving the sum of prior spot counts and the current one
        self.id2name = dict(enumerate(self.names))
        self.flatten = flatten

    def __getitem__(self, index):
        ID = self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1, 0, 2)  # To slice patches aligned with x and y coordinates
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
                patches = torch.zeros((n_patches, patch_dim))  # `flatten` means treating all patches as 1D vectors rather than image grids
            else:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))

            for i in range(n_patches):
                center = centers[i]  # `center` is the center coordinate position
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i] = patch.permute(2, 0, 1)  # Move the channel dimension to the front
            self.patch_dict[ID] = patches
        data = [patches, positions, exps]  # `position` uses (x, y) indices (not pixels); `exps` uses normalized and log-transformed gene expression
        if self.adj:
            data.append(adj)
        if self.ori:
            data += [torch.Tensor(oris), torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        data.append(ID)
        return data  # `data` for one sample: images, positions, and gene expressions

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
        meta = cnt.join((pos.set_index('id')),how='inner')# `set_index` joins by the 'id' column; the resulting `meta` contains both counts (`cnt`) and coordinates
        return meta


def pk_load(fold,mode='train',flatten=False,dataset='her2st',r=2,ori=False,adj=False,prune='Grid',neighs=4):
    '''
    Define an API that directly returns the dataset
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
    trainset = pk_load(0, 'train', False, dataset_name, neighs=4, prune='Grid') # Return the dataset from the API
    train_loader = DataLoader(trainset, batch_size=1, num_workers=10, shuffle=False)   # Batch size must be 1
    for item in train_loader: 
        _id = item[-1][0]
        imgs = item[0].squeeze(0)  #(n,3,224,224)
        exps = item[1].squeeze(0) #(n,785)
        centers = item[-2].squeeze(0)
        centers_dict = {}
        exp_file = {}
        for i_img in range(imgs.shape[0]): # Iterate through the dataloader to save patch-level data
            save_name = _id+'_'+str(i_img)
            centers_dict[save_name] = np.array(centers[i_img]).tolist()
            img = np.array(imgs[i_img].permute(2,1,0)) # Reorder to match cv2's channel order for saving
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert color channels to avoid visualization issues
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