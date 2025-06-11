import os
import torch
import numpy as np
import pandas as pd
from SingleCell_Ref.dataset_utils.my_dataset import Initiate_dataset_with_center
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

def generate_label_prediction_center(model_result_path,samples=None):
   
    model=Sc_Guide_Model_with_CNNSASM('resnet34',785)
   
    patch_path = './01-data/01-her2数据集/01-gen-patch'
    exp_label_path = './01-data/01-her2数据集/02-gene-exp-label/'
    orig_path = './01-data/01-her2数据集/05-orig_exp/'
    center_path = './01-data/01-her2数据集/06-center/'


    if samples==None:
        samples = os.listdir(model_result_path)
        samples.sort()
    for fold in range(len(samples)):
        sample=samples[fold]
        test_index=find_her_index(sample)  

        dataset = Initiate_dataset_with_center(patch_path, exp_label_path, orig_path, center_path, 'test', test_index, 'her2')


        test_sample=dataset.test_sample
        assert sample==test_sample
        test_folder = os.path.join(model_result_path, test_sample)
        dataloder = DataLoader(dataset, batch_size=32, shuffle=False,num_workers=40)

        wts_path = os.path.join(test_folder,test_sample+'-ST_Net-Sc_Guide_Model_with_CNN_SASM-best.pth')
        model.load_state_dict(torch.load(wts_path,map_location='cuda:0'))
        model.to('cuda:0')
        model.eval()

        preds_list = []
        gt_list = []
        ct_list = []
        with torch.no_grad():
            for item in dataloder:

                patch_name, imgs, orig_exp, genes, center = item[0], item[1], item[2], item[3], item[4]
                imgs, orig_exp = imgs.to('cuda:0'), orig_exp.to('cuda:0')
                # pred,_ = model(imgs)  #这个是TCGN的输入
                orig_exp=torch.randn(orig_exp.shape).cuda()
                pred = model(imgs, orig_exp)  #这个是sc-guide的输入
                if len(pred.shape)==1:
                    preds_list.append(pred.unsqueeze(0).cpu().detach().numpy())
                    gt_list.append(genes.detach().numpy())
                    ct_list.append(center.detach().numpy())
                    print('pred_list形状',pred.shape)
                else:
                    preds_list.append(pred.cpu().detach().numpy())
                    gt_list.append(genes.detach().numpy())
                    ct_list.append(center.detach().numpy())

        preds_list = np.concatenate(preds_list, axis=0)
        gt_list = np.concatenate(gt_list, axis=0)
        ct_list = np.concatenate(ct_list, axis=0)

        delete_index=[523,206,181,463]
        preds_list=np.delete(preds_list,delete_index,axis=1)
        gt_list=np.delete(gt_list,delete_index,axis=1)
        save_path = os.path.join(model_result_path,test_sample)
        np.save(os.path.join(save_path,'preds_781.npy'), preds_list)
        np.save(os.path.join(save_path,'gts_781.npy'), gt_list)
        # np.save(os.path.join(save_path,'center.npy'), ct_list)



def compute_pcc_for_all_sample(model_result_path,gene_list_path,save_path):
    gene_list = list(np.load(gene_list_path,allow_pickle=True))
    samples = os.listdir(model_result_path)
    # samples=['A2','A3','A4','A5','A6']
    samples.sort()
    df = pd.DataFrame(columns=gene_list)
    for sample in samples:
        preds_path = os.path.join(model_result_path,sample,'preds_781.npy')
        labels_path = os.path.join(model_result_path,sample,'gts_781.npy')
        preds=np.load(preds_path,allow_pickle=True)
        labels = np.load(labels_path,allow_pickle=True)
        pccs = get_pccs(preds,labels)
        df.loc[sample,:] = pccs

    _mean=[]
    _median=[]
    for gene in range(len(df.columns)):
        _mean.append(np.nanmean(list(df.iloc[:,gene])))
        _median.append(np.nanmedian(list(df.iloc[:,gene])))
    df.loc['mean']=_mean
    df.loc['median']=_median
    df.to_csv(os.path.join(save_path,'消融img-encoder_A_patient_pcc.csv')) 


def calcute_p_after_bh(preds, labels):
    header = list(range(preds[0].shape[0]))
    preds = pd.DataFrame(columns=header, data=preds)
    labels = pd.DataFrame(columns=header, data=labels)
    p_lst = []
    for gene in range(len(preds.columns)):
        _, normal_p_preds = shapiro(preds.iloc[:, gene])
        _, normal_p_label = shapiro(labels.iloc[:, gene])
        if normal_p_label > 0.05 and normal_p_preds > 0.05:
            r, p = pearsonr(preds.iloc[:, gene], labels.iloc[:, gene], alternative='greater')
        else:
            r, p = spearmanr(preds.iloc[:, gene], labels.iloc[:, gene], alternative='greater')
        p_lst.append(p)
    p_bool, p_adjusted = multitest.fdrcorrection(p_lst)
    return p_adjusted  


def compute_p_for_dataset(model_result_path, gene_list_path, save_path):
    gene_list = list(np.load(gene_list_path, allow_pickle=True))
    samples = os.listdir(model_result_path)
    samples.sort()
    df = pd.DataFrame(columns=gene_list)
    for sample in samples:
        preds = np.load(os.path.join(model_result_path, sample, 'preds.npy'), allow_pickle=True)
        labels = np.load(os.path.join(model_result_path, sample, 'gts.npy'), allow_pickle=True)
        p_adjusted = calcute_p_after_bh(preds, labels)
        df.loc[sample, :] = p_adjusted
    _mean = []
    for i in range(len(df.columns)):
        _mean.append(np.nanmean(list(df.iloc[:, i])))
    df.loc['mean'] = _mean
    df.to_csv(os.path.join(save_path, 'p_adjusted.csv'))





if __name__ == '__main__':

    generate_label_prediction_center('./her2/sc-guide/07-random-1timesSc/')

    compute_pcc_for_all_sample(model_result_path,gene_list_path,save_path)
