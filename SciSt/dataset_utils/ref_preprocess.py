import scanpy as sc
import os
import numpy as np
import pandas as pd
import scprep as scp
from collections import Counter


class HER_preprocess_raw_exp():
    def __init__(self, adata, meta_path, gene_path, cell_path, gene_list_path, her=False):
        self.adata = adata
        self.meta = pd.read_csv(meta_path)
        self.adata.var_names = pd.read_csv(cell_path, header=None)[0].values
        self.adata.obs_names = pd.read_csv(gene_path, header=None)[0].values
        self.adata = self.adata.T
        if her:
            meta = pd.read_csv(meta_path)
            her_cell = list(meta[meta.subtype == 'HER2+'].iloc[:, 0])
            adata_her = self.adata[self.adata.obs_names.isin(her_cell), :]
            self.adata = adata_her

        self.anno2Pan_dict = {'inflammatory': ['B-cells', 'T-cells', 'Myeloid'],
                              'connective': ['CAFs', 'Endothelial', 'PVL'],
                              'neoplastic': ['Cancer Epithelial'],
                              'non-neoplastic epithelial': [],
                              'dead': []}
        self.gene_list_path = gene_list_path

    def pick_cells(self, PanNuke_cell_type) -> list:

        if PanNuke_cell_type == 'inflammatory':
            cells = self.meta[self.meta.celltype_major.isin(self.anno2Pan_dict[PanNuke_cell_type])]

            cells = list(cells[cells.celltype_minor != 'Cycling_Myeloid'].iloc[:, 0])
        else:
            cells = list(self.meta[self.meta.celltype_major.isin(self.anno2Pan_dict[PanNuke_cell_type])].iloc[:, 0])
        return cells

    def cells_exp(self, cell_lst):

        adata_subtype = self.adata[self.adata.obs_names.isin(cell_lst), :]
        self.gene_list = self.get_gene_list()
        adata_subtype_df = adata_subtype.to_df()
        adata_subtype_df = adata_subtype_df[self.gene_list]
        return adata_subtype_df

    def compute_mean_exp(self, adata_df):
        return np.mean(np.array(adata_df), axis=0)

    def get_mean_exp(self, cell_type):
        cell_lst = self.pick_cells(cell_type)
        adata_cell_type = self.cells_exp(cell_lst)  # 已经选出了785个基因
        mean_exp = self.compute_mean_exp(adata_cell_type)
        return cell_type, mean_exp

    def get_normalization_exp(self, cell_type, mean_exp):
        mean_exp = mean_exp.reshape(1, mean_exp.shape[0])
        nor_exp = scp.transform.log(scp.normalize.library_size_normalize(mean_exp))
        return cell_type, nor_exp

    def create_df(self):
        self.gene_list = self.get_gene_list()
        breast_df = pd.DataFrame(index=list(self.anno2Pan_dict.keys()), columns=self.gene_list)
        return breast_df

    def get_gene_list(self):
        self.gene_list = np.load(self.gene_list_path)
        need_substitute_genes = ['TESMIN', 'BICDL1', 'GRK3', 'ARHGAP45']
        update_genes = ['MTL5', 'BICD1', 'ADRBK2', 'HMHA1']
        for gene_i in range(len(need_substitute_genes)):
            self.gene_list = self.substitute_gene(self.gene_list, need_substitute_genes[gene_i], update_genes[gene_i])
        return self.gene_list

    def substitute_gene(self, gene_list, orig_gene, update_gene):
        gene_list_update = [update_gene if i == orig_gene else i for i in gene_list]
        return gene_list_update
