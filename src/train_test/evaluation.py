import os 
import torch 
import numpy as np 
import pandas as pd 
from dataset .my_dataset import Initiate_dataset 
from models .sc_guide_model import SciSt
from torch .utils .data import DataLoader 
from utils .evaluation_utils import get_pccs ,calculate_pcc 
from scipy .stats import shapiro 
from scipy .stats import pearsonr ,spearmanr 
from statsmodels .stats import multitest 
from utils .fine_number import find_her_index ,find_cscc_index 

def generate_label_prediction_center (patch_path ,exp_label_path ,orig_path ,model_result_path ,samples =None ):
    model =SciSt ('resnet34',785 ) # Initialize the model
    if samples ==None : # No extra inference sample specified
        samples =os .listdir (model_result_path )
        samples .sort ()
    for fold in range (len (samples )):
        sample =samples [fold ]
        test_index =find_her_index (sample )
        dataset =Initiate_dataset (patch_path ,exp_label_path ,orig_path ,'test',test_index ,'her2') # Load the test dataset

        test_sample =dataset .test_sample 
        assert sample ==test_sample 
        test_folder =os .path .join (model_result_path ,test_sample )
        dataloder =DataLoader (dataset ,batch_size =32 ,shuffle =False ,num_workers =40 )

        wts_path =os .path .join (test_folder ,test_sample +'-SciSt-best.pth') # Load the trained model weights
        model .load_state_dict (torch .load (wts_path ,map_location ='cuda:0'))
        model .to ('cuda:0')
        model .eval () # Switch to evaluation mode

        preds_list =[] # Save inference results
        gt_list =[] # Save ground truth (optional: can load directly from ./02-label)

        with torch .no_grad ():
            for item in dataloder :
                patch_name ,imgs ,orig_exp ,genes ,center =item [0 ],item [1 ],item [2 ],item [3 ],item [4 ]
                imgs ,orig_exp =imgs .to ('cuda:0'),orig_exp .to ('cuda:0')
                pred =model (imgs ,orig_exp ) # This is the input format for sc-guide
                if len (pred .shape )==1 : # Different models may output different shapes for evaluation, e.g., (1,785) vs (785,)
                    preds_list .append (pred .unsqueeze (0 ).cpu ().detach ().numpy ())
                    gt_list .append (genes .detach ().numpy ())
                    print ('pred_list shape',pred .shape )
                else :
                    preds_list .append (pred .cpu ().detach ().numpy ())
                    gt_list .append (genes .detach ().numpy ())


        preds_list =np .concatenate (preds_list ,axis =0 ) # Concatenate predictions for each patch and save them together
        gt_list =np .concatenate (gt_list ,axis =0 )

          # Delete_index=[523,206,181,463]#These four genes have zero predictions; p-values cannot be computed, so 781 genes are included in evaluation.
        # preds_list=np.delete(preds_list,delete_index,axis=1)
        # gt_list=np.delete(gt_list,delete_index,axis=1)
        save_path =os .path .join (model_result_path ,test_sample )
        np .save (os .path .join (save_path ,'preds.npy'),preds_list )
        np .save (os .path .join (save_path ,'gts.npy'),gt_list )


def compute_pcc_for_all_sample (model_result_path ,gene_list_path ,save_path ):
    gene_list =list (np .load (gene_list_path ,allow_pickle =True ))  # Load the predicted gene names
    samples =os .listdir (model_result_path )
    # samples=['A2','A3','A4','A5','A6']
    samples .sort ()
    df =pd .DataFrame (columns =gene_list )  # Save each PCC to a CSV file: rows are samples and columns are predicted genes
    for sample in samples :
        preds_path =os .path .join (model_result_path ,sample ,'preds.npy')  # Load inference results
        labels_path =os .path .join (model_result_path ,sample ,'gts.npy')  # Load ground-truth results
        preds =np .load (preds_path ,allow_pickle =True )
        labels =np .load (labels_path ,allow_pickle =True )
        pccs =get_pccs (preds ,labels )  # Use the get_pccs API to compute the PCC list
        df .loc [sample ,:]=pccs 

    _mean =[]  # Save the mean PCC of each gene across all samples
    _median =[]  # Save the median PCC of each gene across all samples
    for gene in range (len (df .columns )):
        _mean .append (np .nanmean (list (df .iloc [:,gene ])))
        _median .append (np .nanmedian (list (df .iloc [:,gene ])))
    df .loc ['mean']=_mean 
    df .loc ['median']=_median 
    df .to_csv (os .path .join (save_path ,'pcc.csv'))


def calcute_p_after_bh (preds ,labels ):
    header =list (range (preds [0 ].shape [0 ]))
    preds =pd .DataFrame (columns =header ,data =preds )
    labels =pd .DataFrame (columns =header ,data =labels )
    p_lst =[]
    for gene in range (len (preds .columns )):
        _ ,normal_p_preds =shapiro (preds .iloc [:,gene ])  # Test whether preds follow a normal distribution
        _ ,normal_p_label =shapiro (labels .iloc [:,gene ])  # Test whether labels follow a normal distribution
        if normal_p_label >0.05 and normal_p_preds >0.05 :  # If both are normally distributed, use pearsonr
            r ,p =pearsonr (preds .iloc [:,gene ],labels .iloc [:,gene ],alternative ='greater')  # one-sided test
        else :  # Otherwise, use spearmanr
            r ,p =spearmanr (preds .iloc [:,gene ],labels .iloc [:,gene ],alternative ='greater')
        p_lst .append (p )  # Save the p-value for each gene
    p_bool ,p_adjusted =multitest .fdrcorrection (p_lst )  # Apply BH correction to all p-values
    return p_adjusted   # Save BH-corrected p-values for each gene


def compute_p_for_dataset (model_result_path ,gene_list_path ,save_path ):
    gene_list =list (np .load (gene_list_path ,allow_pickle =True ))
    samples =os .listdir (model_result_path )
    samples .sort ()
    df =pd .DataFrame (columns =gene_list )
    for sample in samples :
        preds =np .load (os .path .join (model_result_path ,sample ,'preds.npy'),allow_pickle =True )
        labels =np .load (os .path .join (model_result_path ,sample ,'gts.npy'),allow_pickle =True )
        p_adjusted =calcute_p_after_bh (preds ,labels )  # Call the BH-corrected p-value calculation for a single sample
        df .loc [sample ,:]=p_adjusted   # Save as a CSV file
    _mean =[]  # Compute the mean p-value of each gene across all samples
    for i in range (len (df .columns )):
        _mean .append (np .nanmean (list (df .iloc [:,i ])))
    df .loc ['mean']=_mean 
    df .to_csv (os .path .join (save_path ,'p_adjusted.csv'))



if __name__ =='__main__':

    patch_path ='./01_patch/'
    exp_label_path ='./02_label/'
    orig_path ='./06-noise_exp/'
    model_result_path ='./07-SciSt/'
     # Inference
    generate_label_prediction_center (patch_path ,exp_label_path ,orig_path ,model_result_path )

      # Save PCCs and p-value results
    gene_list_path ='./hvg_1000.npy'
    save_path ='./results/'
    compute_pcc_for_all_sample (model_result_path ,gene_list_path ,save_path )
    compute_p_for_dataset (model_result_path ,gene_list_path ,save_path )
