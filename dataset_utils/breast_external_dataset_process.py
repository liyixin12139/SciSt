from PIL import ImageFile, Image
import numpy as np
import pandas as pd
import tifffile as tf
import cv2
import os
import json
import scprep as scp
import torch
import scanpy as sc
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from torch.utils.data import DataLoader,Dataset


def convert_image(input_filename, output_filename, downsample):
    """Converts an OME TIFF image from RGB to BGR and downsamples it."""
    with tf.TiffFile(input_filename) as tif:
        main_image = tif.asarray()
        if main_image.shape[0] == 3:
            width, height = main_image.shape[2], main_image.shape[1]
            main_image = np.transpose(main_image, (1, 2, 0))
        else:
            width, height = main_image.shape[1], main_image.shape[0]

        image_bgr = cv2.cvtColor(main_image, cv2.COLOR_RGB2BGR)
        image_bgr = cv2.resize(image_bgr, (width // downsample, height // downsample))

        cv2.imwrite(output_filename, image_bgr)
        print(f"Image saved to {output_filename}")

class Breast_External_ST(Dataset):
    def __init__(self):
        super().__init__()
        self.img_path = './Visium_FFPE_Human_Breast_Cancer_image.tif'
        self.pos_path = './spatial/tissue_positions_list.csv'
        self.exp_path = './Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5'
        gene_list_path = './01-data/02-乳腺癌/breast_train_test_gene_list.npy'
        gene_list = list(np.load(gene_list_path,allow_pickle=True))  
        half_pixel_size_for_spot = int(112)
        self.gene_set = gene_list
        self.exps = self.get_exp()  
        with tf.TiffFile(self.img_path) as tif:
            img = tif.asarray()

        height, width, _ = img.shape
        self.pos,spot_lst = self.get_pos()


        self.exp_list = {spot: scp.transform.log(scp.normalize.library_size_normalize(np.array(self.exps.loc[spot,:]).reshape(1,len(self.gene_set)))) for spot in spot_lst}
        self.pos_dict={spot:self.pos.loc[spot,:][['X','Y']].values for spot in spot_lst}
        self.center_dict={spot:np.floor(self.pos.loc[spot,:][['pixel_x','pixel_y']].values).astype(int) for spot in spot_lst}
        self.patch_list = {}
        self.spot_names = spot_lst
        for spot in spot_lst:
            x_center = self.center_dict[spot][0]
            y_center = self.center_dict[spot][1]
            patch = torch.Tensor(img[x_center-half_pixel_size_for_spot:x_center+half_pixel_size_for_spot,y_center-half_pixel_size_for_spot:y_center+half_pixel_size_for_spot,:])
            patch = patch.permute(2, 0, 1)
            self.patch_list[spot] = patch

    def __getitem__(self, item):
        spot_name = self.spot_names[item]
        img = self.patch_list[spot_name]
        gene = self.exp_list[spot_name]
        center=self.center_dict[spot_name]
        pos=self.pos_dict[spot_name]
        return (spot_name,img,gene,pos,center)

    def __len__(self):
        return len(self.spot_names)
    def get_exp(self):
        adata = sc.read_10x_h5(self.exp_path)
        adata.var_names_make_unique()
        adata_df = adata.to_df()
        exp = adata_df[self.gene_set]
        return exp
    def get_pos(self):
        pos = pd.read_csv(self.pos_path, header=None)
        pos = pos[pos[1] == 1]
        pos.index = pos.iloc[:, 0]
        pos = pos.iloc[:, 2:]
        pos.columns = ['X', 'Y', 'pixel_x', 'pixel_y']
        spot_lst=list(pos.index)
        return pos,spot_lst


def generate_patch_label():
    dataset = Breast_External_ST()
    data_loader = DataLoader(dataset, batch_size=1)
    save_path = './01-data/02-乳腺癌/07-验证的breast-tiff/01-gen-patch/'
    gene_save_path = './01-data/02-乳腺癌/07-验证的breast-tiff/02-gene-exp-label/'
    exp_file = {}
    for item in data_loader:
        _id = str(item[0][0])
        imgs = item[1].squeeze(0)  #(n,3,224,224)
        exps = item[2].squeeze(0) #(n,183)
        img = np.array(imgs.permute(2,1,0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, _id + '.png'), img)
        exp_file[_id] = np.array(exps).squeeze().tolist()
    with open(os.path.join(gene_save_path,'liver_gene.json'),'w') as f:
        json.dump(exp_file,f)


#将xenium的micron转化为pixel
if __name__ == '__main__':

    generate_patch_label()

