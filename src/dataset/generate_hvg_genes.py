import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc

if __name__ == "__main__":
    #Set common params
    parser = argparse.ArgumentParser()
    parser.add_argument( '--h5_path', type=str, help='H5 file path', required=True)
    parser.add_argument( '--save_path', type=str, required=True)
    args = parser.parse_args()
    
    path = args.h5_path
    liver_adata = sc.read_10x_h5(path) #load ST data
    liver_adata.obs_names_make_unique()
    liver_adata.var_names_make_unique()
    sc.pp.normalize_total(liver_adata) #normalize exp data
    sc.pp.log1p(liver_adata) #log-transform
    sc.pp.highly_variable_genes(liver_adata, n_top_genes=1000,subset=True) #choose top 1000 high variable genes. Change it if you need 
    sc.pp.filter_genes(liver_adata, min_cells=1000) #filter genes which expression less than 1000 cells. Change it if you need
    liver_hvg_cut_1000 = list(liver_adata.var_names)
    np.save(os.path.join(args.save_path,'liver_hvg_1000.npy'),liver_hvg_cut_1000)