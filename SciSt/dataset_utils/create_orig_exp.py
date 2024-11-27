import os
import json
import scprep as scp
import numpy as np
import pandas as pd


def count_celltype(json_path,id2tissue_dict)->dict:
    '''
    从hovernet的分割结果json文件中整理出不同细胞类型的计数
    :param json_path: .json文件
    :param id2tissue_dict: dict
    '''
    with open(json_path,'r') as f:
        json_file = json.load(f)
    json_nuc_dict = json_file['nuc']

    count_dict = {}
    for celltype in list(id2tissue_dict.values()):
        count_dict[celltype] = 0

    for cell_i in json_nuc_dict.values():
        sin_celltype = id2tissue_dict[str(cell_i['type'])]
        count_dict[sin_celltype]+=1
    f.close()
    return count_dict


def substitute_gene(gene_list,orig_gene,update_gene):
    gene_list_update = [update_gene if i==orig_gene else i for i in gene_list]
    return gene_list_update

def process_ref(ref_path)->pd.DataFrame:
    '''
    处理ref文件，需要将基因名设置为行名，且转置，形状为（cell_num,gene_num）
    :param ref_path: csv文件
    :return:
    '''
    ref = pd.read_csv(ref_path)
    ref.set_index('Unnamed: 0', inplace=True)
    return ref


def create_orig_exp(count_dict,sincell_ref):
    weight_dict = {}
    her2_ref_celltype = ['neoplastic', 'inflammatory', 'connective','non-neoplastic epithelial']  
    ref_sum = sum([v for k, v in count_dict.items() if k in her2_ref_celltype])
    orig_exp = np.zeros((1,sincell_ref.shape[1]))
    if ref_sum==0:
        for celltype in her2_ref_celltype:
            orig_exp += np.array(sincell_ref[sincell_ref.index==celltype])*(1/len(her2_ref_celltype))
        return orig_exp
    else:
        for k, v in count_dict.items():
            if k in her2_ref_celltype:
                weight_dict[k] = count_dict[k] / ref_sum
        for k,v in weight_dict.items():
            orig_exp += np.array(sincell_ref[sincell_ref.index==k])*weight_dict[k]
        return orig_exp

def create_orig_exp_for_patch(json_path,predicted_gene_path,ref_path):
    id2tissue = {'0':'nolabe',
                 '1':'neoplastic',
                 '2':'inflammatory',
                 '3':'connective',
                 '4':'dead',
                 '5':'non-neoplastic epithelial'}
    result = count_celltype(json_path,id2tissue)
    predicted_genes = list(np.load(predicted_gene_path, allow_pickle=True))
    ref = process_ref(ref_path)

    need_substitute_genes = ['TESMIN','BICDL1','GRK3','ARHGAP45']
    update_genes = ['MTL5','BICD1','ADRBK2','HMHA1']
    for gene_i in range(len(need_substitute_genes)):
        predicted_genes = substitute_gene(predicted_genes,need_substitute_genes[gene_i],update_genes[gene_i])
    ref = ref[predicted_genes] #（cell_num,gene_num）在这里是（5，785）
    orig_exp = create_orig_exp(result,ref)
    return orig_exp

def her_create_orig_exp_for_patch_scp(json_path,predicted_gene_path,ref_path):

    id2tissue = {'0':'nolabe',
                 '1':'neoplastic',
                 '2':'inflammatory',
                 '3':'connective',
                 '4':'dead',
                 '5':'non-neoplastic epithelial'}
    result = count_celltype(json_path,id2tissue)

    predicted_genes = list(np.load(predicted_gene_path, allow_pickle=True))

    #整理ref文件
    ref = process_ref(ref_path)

    need_substitute_genes = ['TESMIN','BICDL1','GRK3','ARHGAP45']
    update_genes = ['MTL5','BICD1','ADRBK2','HMHA1']
    for gene_i in range(len(need_substitute_genes)):
        predicted_genes = substitute_gene(predicted_genes,need_substitute_genes[gene_i],update_genes[gene_i])
    ref = ref[predicted_genes] #（cell_num,gene_num）在这里是（5，785）
    ref_scp = np.array(ref.iloc[:3,:])
    ref_scp = scp.transform.log(scp.normalize.library_size_normalize(ref_scp))
    for i in range(ref_scp.shape[0]):
        ref.iloc[i,:] = ref_scp[i]
    orig_exp = create_orig_exp(result,ref)
    return orig_exp

def gen_origExp_for_all_patch(gene_num):
    #对每张patch生成orig_exp.npy
    path = './01-data/02-乳腺癌/06-TCGA/02-WSI_patch/04-TCGA-PL-A8LZ/02-segResult/json/'
    ref_path = './01-data/01-her2数据集/04-singlecell_ref/small_num_785_mean_scp.csv'
    save_path = './01-data/02-乳腺癌/06-TCGA/02-WSI_patch/04-TCGA-PL-A8LZ/07-785-orig-exp/'
    id2tissue = {'0':'nolabe',
                 '1':'neoplastic',
                 '2':'inflammatory',
                 '3':'connective',
                 '4':'dead',
                 '5':'non-neoplastic epithelial'}
    ref = process_ref(ref_path)
    for json_file in os.listdir(path):
        spec_path = os.path.join(path,json_file)
        result = count_celltype(spec_path,id2tissue)

        orig_exp = create_orig_exp(result,ref)
        orig_exp = orig_exp.reshape((gene_num,))
        np.save(os.path.join(save_path,json_file.split('.')[0]+'.npy'),orig_exp)




if __name__ == '__main__':

    gen_origExp_for_all_patch(785)


