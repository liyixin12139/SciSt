import os
import torch
import numpy as np
import pandas as pd
from SingleCell_Ref.dataset_utils.my_dataset import Initiate_dataset
from SingleCell_Ref.singleCell_models.sc_guide_model import Sc_Guide_Model_with_CNNSASM,Sc_Guide_Model_with_Resnet34,Concat_Model_for_Ablation
from torch.utils.data import DataLoader
from Paper_THItoGene.THItoGene_dataset.thitogene_dataset import ViT_HER2ST,ViT_SKIN
from Paper_THItoGene.THItoGene_model.vis_model import THItoGene
from paperTCGN.TCGN_model.TCGN import TCGN
from SingleCell_Ref.utils.evaluation_utils import get_pccs,calculate_pcc
from scipy.stats import shapiro
from scipy.stats import pearsonr,spearmanr
from statsmodels.stats import multitest
from SingleCell_Ref.utils.fine_number import find_her_index,find_cscc_index

def generate_label_prediction_center(patch_path,exp_label_path,orig_path,model_result_path,samples=None):
    model=Sc_Guide_Model_with_CNNSASM('resnet34',785) #初始化模型
    if samples==None: #不额外定义推理的sample
        samples = os.listdir(model_result_path)
        samples.sort()
    for fold in range(len(samples)):
        sample=samples[fold]
        test_index=find_her_index(sample)  
        dataset = Initiate_dataset(patch_path, exp_label_path, orig_path,  'test', test_index, 'her2') #加载test的dataset

        test_sample=dataset.test_sample
        assert sample==test_sample
        test_folder = os.path.join(model_result_path, test_sample)
        dataloder = DataLoader(dataset, batch_size=32, shuffle=False,num_workers=40)

        wts_path = os.path.join(test_folder,test_sample+'-SciSt-best.pth') #加载训练好的模型权重
        model.load_state_dict(torch.load(wts_path,map_location='cuda:0'))
        model.to('cuda:0')
        model.eval() #进入评估模式

        preds_list = [] # 保存推理结果
        gt_list = [] #保存真实结果 也可以不保存，直接加载./02-label

        with torch.no_grad():
            for item in dataloder:
                patch_name, imgs, orig_exp, genes, center = item[0], item[1], item[2], item[3], item[4]
                imgs, orig_exp = imgs.to('cuda:0'), orig_exp.to('cuda:0')
                pred = model(imgs, orig_exp)  #这个是sc-guide的输入
                if len(pred.shape)==1: #不同模型评估输出形状不一样，(1,785)/(785,)
                    preds_list.append(pred.unsqueeze(0).cpu().detach().numpy())
                    gt_list.append(genes.detach().numpy())
                    print('pred_list形状',pred.shape)
                else:
                    preds_list.append(pred.cpu().detach().numpy())
                    gt_list.append(genes.detach().numpy())


        preds_list = np.concatenate(preds_list, axis=0) #每个patch的预测结果做拼接，一起保存
        gt_list = np.concatenate(gt_list, axis=0)

        # delete_index=[523,206,181,463]#这四个基因预测有0，后续无法计算p value，因此纳入evaluate的是781个基因
        # preds_list=np.delete(preds_list,delete_index,axis=1)
        # gt_list=np.delete(gt_list,delete_index,axis=1)
        save_path = os.path.join(model_result_path,test_sample)
        np.save(os.path.join(save_path,'preds.npy'), preds_list)
        np.save(os.path.join(save_path,'gts.npy'), gt_list)


def compute_pcc_for_all_sample(model_result_path,gene_list_path,save_path):
    gene_list = list(np.load(gene_list_path,allow_pickle=True)) #加载预测的基因名
    samples = os.listdir(model_result_path)
    # samples=['A2','A3','A4','A5','A6']
    samples.sort()
    df = pd.DataFrame(columns=gene_list) #将每个pcc保存成一个csv文件，行为sample，列为预测基因
    for sample in samples:
        preds_path = os.path.join(model_result_path,sample,'preds.npy') #加载推理结果
        labels_path = os.path.join(model_result_path,sample,'gts.npy') #加载真实结果
        preds=np.load(preds_path,allow_pickle=True)
        labels = np.load(labels_path,allow_pickle=True)
        pccs = get_pccs(preds,labels) #用get_pccs api计算pcc list
        df.loc[sample,:] = pccs

    _mean=[] #保存每个基因在所有样本上的mean pcc
    _median=[] #保存每个基因在所有样本上的median pcc
    for gene in range(len(df.columns)):
        _mean.append(np.nanmean(list(df.iloc[:,gene])))
        _median.append(np.nanmedian(list(df.iloc[:,gene])))
    df.loc['mean']=_mean
    df.loc['median']=_median
    df.to_csv(os.path.join(save_path,'pcc.csv')) 


def calcute_p_after_bh(preds, labels):
    header = list(range(preds[0].shape[0]))
    preds = pd.DataFrame(columns=header, data=preds)
    labels = pd.DataFrame(columns=header, data=labels)
    p_lst = []
    for gene in range(len(preds.columns)):
        _, normal_p_preds = shapiro(preds.iloc[:, gene]) #检验preds是否满足正态分布
        _, normal_p_label = shapiro(labels.iloc[:, gene]) #检验labels是否满足正态分布
        if normal_p_label > 0.05 and normal_p_preds > 0.05: #如果两个都满足正态分布，就用pearsonr检验
            r, p = pearsonr(preds.iloc[:, gene], labels.iloc[:, gene], alternative='greater') #单侧检验
        else: #否则用spearmanr检验
            r, p = spearmanr(preds.iloc[:, gene], labels.iloc[:, gene], alternative='greater')
        p_lst.append(p) #保存每个基因的p值
    p_bool, p_adjusted = multitest.fdrcorrection(p_lst) #为所有p值统一做bh矫正
    return p_adjusted   #保存bh矫正后每个基因的p值


def compute_p_for_dataset(model_result_path, gene_list_path, save_path):
    gene_list = list(np.load(gene_list_path, allow_pickle=True))
    samples = os.listdir(model_result_path)
    samples.sort()
    df = pd.DataFrame(columns=gene_list)
    for sample in samples:
        preds = np.load(os.path.join(model_result_path, sample, 'preds.npy'), allow_pickle=True)
        labels = np.load(os.path.join(model_result_path, sample, 'gts.npy'), allow_pickle=True)
        p_adjusted = calcute_p_after_bh(preds, labels) #调用单个sample的bh矫正p值计算结果
        df.loc[sample, :] = p_adjusted #保存为一个csv文件
    _mean = [] #计算每个基因在所有样本的mean p值
    for i in range(len(df.columns)):
        _mean.append(np.nanmean(list(df.iloc[:, i])))
    df.loc['mean'] = _mean
    df.to_csv(os.path.join(save_path, 'p_adjusted.csv'))



if __name__ == '__main__':

    patch_path = './01_patch/'
    exp_label_path = './02_label/'
    orig_path = './06-noise_exp/'
    model_result_path='./07-SciSt/'
    #推理
    generate_label_prediction_center(patch_path,exp_label_path,orig_path,model_result_path)

    #保存pcc和p值结果
    gene_list_path='./hvg_1000.npy'
    save_path='./results/'
    compute_pcc_for_all_sample(model_result_path,gene_list_path,save_path)
    compute_p_for_dataset(model_result_path,gene_list_path,save_path)
