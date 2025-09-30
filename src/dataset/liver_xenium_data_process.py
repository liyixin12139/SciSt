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


def read_transformation_matrix(filepath):
    """
    读取的是alignment file.csv
    Reads and returns the transformation matrix from a CSV file."""
    try:
        return pd.read_csv(filepath, header=None).values
    except Exception as e:
        raise IOError(f"Error reading transformation matrix: {e}")


def transform_coordinates(df, multiplier, transformation_matrix, downsample):
    """Applies transformation matrix to the coordinates and adjusts for downsampling."""
    df['pixel_x'] = df['x_centroid'] * multiplier
    df['pixel_y'] = df['y_centroid'] * multiplier

    transformed = np.dot(np.linalg.inv(transformation_matrix),
                         np.vstack([df['pixel_x'], df['pixel_y'], np.ones(len(df))]))

    df['pixel_x'] = (transformed[0, :] // downsample).astype(int)
    df['pixel_y'] = (transformed[1, :] // downsample).astype(int)
    return df


def group_cells(df, patch_size, output_folder):
    """Groups cells based on their transformed coordinates and patch size."""
    df['group_x'] = (df['pixel_x'] // patch_size).astype(int)
    df['group_y'] = (df['pixel_y'] // patch_size).astype(int)
    df['group_name'] = df['group_x'].astype(str) + '_' + df['group_y'].astype(str)
    output_file_path = os.path.join(output_folder, "grouped_cells.csv")
    df.to_csv(output_file_path, index=False)
    print(f'Grouped cells saved to {output_file_path}')


def transform_coordinates_and_group(csv_file, multiplier, patch_size, output_folder, aligned, transformation_matrix_dir,
                                    downsample):
    """Main function to process cell data, transform coordinates and group cells."""
    df = pd.read_csv(csv_file)
    if not aligned:
        transformation_matrix = read_transformation_matrix(transformation_matrix_dir)
    else:
        transformation_matrix = np.eye(3)
    transformed = transform_coordinates(df, multiplier, transformation_matrix, downsample)
    group_cells(transformed, patch_size, output_folder)


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

class Liver_ST(Dataset):
    def __init__(self):
        super().__init__()
        self.img_path = './14-肝癌验证-数据来自10XVisium in situ gene expression/肝癌/33795_15404.png'
        self.pos_path = './14-肝癌验证-数据来自10XVisium in situ gene expression/肝癌/grouped_cells.csv'
        self.exp_path = './14-肝癌验证-数据来自10XVisium in situ gene expression/肝癌/output-Xenium_V1_hLiver_cancer_section_FFPE_outs/cell_feature_matrix.h5'
        gene_list_path = './14-肝癌验证-数据来自10XVisium in situ gene expression/肝癌/liver_hvg_cut_200_minus3.npy'
        gene_list = list(np.load(gene_list_path,allow_pickle=True))
        half_pixel_size_for_spot = int(112)
        self.gene_set = gene_list
        self.exps = self.get_exp()  #
        img = torch.Tensor(np.array(self.get_img())) #(33795, 15404, 3)
        height, width, _ = img.shape
        self.pos = self.get_pos()
        self.grouped = self.pos.groupby('group_name')

        self.exp_list = {group_name: scp.transform.log(scp.normalize.library_size_normalize(self.get_group_exp(group_name))) for group_name,group_data in self.grouped}
        self.patch_list = {}
        self.spot_names = []
        for group_name,group_data in self.grouped:
            x_center = int(np.mean(list(group_data['pixel_x'])))
            y_center = int(np.mean(list(group_data['pixel_y'])))
            patch = img[y_center-half_pixel_size_for_spot:y_center+half_pixel_size_for_spot,x_center-half_pixel_size_for_spot:x_center+half_pixel_size_for_spot,:]
            patch = patch.permute(2, 0, 1)
            self.patch_list[group_name] = patch
            self.spot_names.append(group_name)

    def __getitem__(self, item):
        spot_name = self.spot_names[item]
        img = self.patch_list[spot_name]
        gene = self.exp_list[spot_name]
        return (spot_name,img,gene)

    def __len__(self):
        return len(self.spot_names)
    def get_img(self):
        img = Image.open(self.img_path)#(w,h,3)
        return img
    def get_exp(self):
        adata = sc.read_10x_h5(self.exp_path)
        adata_df = adata.to_df()
        exp = adata_df[self.gene_set]
        return exp
    def get_pos(self):
        df = pd.read_csv(self.pos_path).iloc[:, 1:]  
        df.set_index('cell_id', inplace=True)
        return df
    def get_group_exp(self,group_name):
        group_data = self.grouped.get_group(group_name)
        group_cell_ids = list(group_data.index)
        group_cell_exps = self.exps.loc[group_cell_ids]
        group_exp = np.sum(group_cell_exps.values,axis=0,keepdims=True) #(1,183)
        return group_exp

def generate_patch_label():
    dataset = Liver_ST()
    data_loader = DataLoader(dataset, batch_size=1)
    save_path = './01-data/04-Liver/01-gen-patch/'
    gene_save_path = './01-data/04-Liver/02-gene-exp-label/'
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

    #1. 从centroid生成pixel的df
    # cell_path = './14-肝癌验证-数据来自10XVisium in situ gene expression/肝癌/output-Xenium_V1_hLiver_cancer_section_FFPE_outs/cells.csv'
    # alignment_path = './14-肝癌验证-数据来自10XVisium in situ gene expression/肝癌/output-Xenium_V1_hLiver_cancer_section_FFPE_outs/Xenium_V1_hLiver_cancer_section_FFPE_he_imagealignment.csv'
    # df = pd.read_csv(cell_path)
    # transformation_matrix = pd.read_csv(alignment_path, header=None).values
    # multipier = 1 / 0.2125
    # df = transform_coordinates(df, multipier, transformation_matrix, 1)
    # df.to_csv(
    #     './14-肝癌验证-数据来自10XVisium in situ gene expression/肝癌/pixel_orridinate_downsample1.csv')


    # input_file_name = './14-肝癌验证-数据来自10XVisium in situ gene expression/肝癌/input-Xenium_V1_hLiver_cancer_section_FFPE_he_image.ome.tif'
    # output_filename = './14-肝癌验证-数据来自10XVisium in situ gene expression/肝癌/33795_15404.png'
    # convert_image(input_file_name,output_filename,1)



    #3. 计算每个细胞对应的spot，按照x_y的格式,使用group-cells函数
    # df = pd.read_csv('./14-肝癌验证-数据来自10XVisium in situ gene expression/肝癌/pixel_orridinate_33795_15404.csv')
    # patch_size=168
    # output_folder = './14-肝癌验证-数据来自10XVisium in situ gene expression/肝癌/'
    # group_cells(df,patch_size,output_folder)

    #生成patch和label
    generate_patch_label()



