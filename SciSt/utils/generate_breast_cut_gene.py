import anndata as ad
import os
import numpy as np
import scanpy as sc
def adata_process(path,index):
    import scanpy as sc
    adata = sc.read_text(path)
    picked_genes = [i for i in adata.var_names if not i.startswith('__ambiguous')]
    adata_picked_genes = adata[:,picked_genes]
    adata_picked_genes.obs_names = [str(index)+'_'+i for i in adata_picked_genes.obs_names]
    adata_picked_genes.obs_names_make_unique()
    adata_picked_genes.var_names_make_unique()
    return adata_picked_genes
def integrate_multi_adata(tsv_path):
    adata_list = []
    i = 0
    for file in os.listdir(tsv_path):
        file_path = os.path.join(tsv_path,file)
        adata_picked_genes = adata_process(file_path,i)
        i+=1
        adata_list.append(adata_picked_genes)
        print(i,': ',file)
    adata_merge = ad.concat(adata_list,merge='same')
    sc.pp.normalize_total(adata_merge)
    sc.pp.log1p(adata_merge)
    sc.pp.highly_variable_genes(adata_merge, n_top_genes=1000,subset=True)
    sc.pp.filter_genes(adata_merge, min_cells=1000)
    return list(adata_merge.var_names)
print('start...')
tsv_path = './5-STNet-Human breast cancer in situ capturing transcriptomics/cnt_dir_orig/'
gene_list = integrate_multi_adata(tsv_path)
np.save('./5-STNet-Human breast cancer in situ capturing transcriptomics/breast_hvg_cut_1000.npy',gene_list)
print('end')