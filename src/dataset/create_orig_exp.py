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
    json_nuc_dict = json_file['nuc'] #获取分类标签

    count_dict = {}
    for celltype in list(id2tissue_dict.values()): #加细胞类型到字典key里
        count_dict[celltype] = 0

    for cell_i in json_nuc_dict.values():
        sin_celltype = id2tissue_dict[str(cell_i['type'])]
        count_dict[sin_celltype]+=1 #对patch中每个细胞类型计数
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
    '''
    根据hovernet的分割结果，各种细胞类型计数，从ref中按照权重计算初始表达
    根据不同数据集计算方式应该是不同的，所以后续需要写成class，先写一个her2数据集的函数
    :param count_dict:
    :param sincell_ref:
    :return:
    '''
    weight_dict = {}
    ref_celltype = ['neoplastic', 'inflammatory', 'connective','non-neoplastic epithelial']  
    ref_sum = sum([v for k, v in count_dict.items() if k in ref_celltype]) #对patch的细胞总数做记录
    orig_exp = np.zeros((1,sincell_ref.shape[1])) #存放输出结果
    if ref_sum==0: #如果patch上没有检测到任何细胞
        for celltype in ref_celltype:
            orig_exp += np.array(sincell_ref[sincell_ref.index==celltype])*(1/len(ref_celltype)) #对ref的每个细胞类型表达取均值
        return orig_exp
    else:
        for k, v in count_dict.items():
            if k in ref_celltype:
                weight_dict[k] = count_dict[k] / ref_sum #根据每个细胞类型出现数目/总数目 记录每个细胞类型权重
        for k,v in weight_dict.items():
            orig_exp += np.array(sincell_ref[sincell_ref.index==k])*weight_dict[k] #每个ref表达✖️对应权重，得到noise-exp
        return orig_exp

def gen_origExp_for_all_patch(path,ref_path,save_path,add_noise=False):
    id2tissue = {'0':'nolabe', #hover-net的分类标签
                 '1':'neoplastic',
                 '2':'inflammatory',
                 '3':'connective',
                 '4':'dead',
                 '5':'non-neoplastic epithelial'}
    ref = process_ref(ref_path)
    gene_num=len(ref.columns)
    for json_file in os.listdir(path): #每个patch依次处理
        spec_path = os.path.join(path,json_file)
        result = count_celltype(spec_path,id2tissue)
        ####可以在cellratio上加噪声，验证robustness
        if add_noise:
            noise_range = (-2, 2)  # 调整噪声范围，示例中是[-2, 2]的范围
            noisy_cell_types = {}
            for key, value in result.items():
                # 随机生成噪声
                noise = random.randint(noise_range[0], noise_range[1])
                noisy_value = value + noise
                noisy_cell_types[key] = max(0, noisy_value)
            result=noisy_cell_types
        # #整理单细胞ref文件，需要从ref中找出空间转录组有表达的基因，在HER2+数据集中有785个基因
        orig_exp = create_orig_exp(result,ref)
        orig_exp = orig_exp.reshape((gene_num,))
        np.save(os.path.join(save_path,json_file.split('.')[0]+'.npy'),orig_exp)




if __name__ == '__main__':
    path = './04-patch-segResult/json/' #Hover-Net outputs path
    ref_path = './05-ref.csv' #ref path
    save_path = './06-noise_exp/' #save path
    gen_origExp_for_all_patch(path,ref_path,save_path)


