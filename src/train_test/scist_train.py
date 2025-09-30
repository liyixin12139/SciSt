import os 
import numpy as np 
import pandas as pd 
import torch 
import torch .nn as nn 
import yaml 
import argparse 
import datetime 
import gc 
import random 
import warnings 
from torch .utils .data import DataLoader 
from dataset .my_dataset import load_dataset 
from paperTCGN .TCGN_model .TCGN import TCGN 
from utils .draw_loss_pcc import draw_loss ,draw_pcc 
from models .sc_guide_model import get_model 
from utils .evaluation_utils import compare_prediction_label_list 


use_gpu =torch .cuda .is_available () # GPU acceleration
#use_gpu=False
torch .cuda .empty_cache () # Clear CUDA cache

 # Fix random seed，ensure reproducibility
def seed_torch (seed =0 ):
    random .seed (seed )
    os .environ ['PYTHONHASHSEED']=str (seed )
    np .random .seed (seed )
    torch .manual_seed (seed )
    torch .cuda .manual_seed (seed )
    torch .cuda .manual_seed_all (seed )
    torch .backends .cudnn .benchmark =False 
    torch .backends .cudnn .deterministic =True 
seed_torch ()

def scist_train (config ):
    batch_size =config ['batch_size']
    epoch =config ['epoch']
    starttime =datetime .datetime .now ().strftime ('%Y-%m-%d %H:%M:%S')
    print ("=========="*8 +"%s"%starttime )
    print ("GPU available:",use_gpu )
    # load data
    train_loader ,test_loader ,test_sample =load_dataset (config ) # Return dataloaders from the API
    print ("finish loading")
    print (test_sample )

    # initialize model
    import os 
    model_name =config ['model_name']
    output_path =os .path .join (config ['output_path'],test_sample )
    if not os .path .exists (output_path ):
        os .makedirs (output_path ) # Used to save trained models
    if model_name !='SciSt'and model_name !='SciSt_Ablation': # SciSt and other models require different input arguments
        my_model =get_model (config ['model_name'])()
    else :
        my_model =get_model (config ['model_name'])(config ['img_encoder'],config ['num_gene'])

    if use_gpu :
        my_model =my_model .cuda ()
    if config ['wts_path']!=None :  # If weights are provided, load them entirely
        my_model .load_state_dict (torch .load (config ['wts_path']),strict =False ) # This should be a pretrained model from the timm package

        # train the model
    optimizer =torch .optim .Adam (my_model .parameters (),lr =1e-5 ,betas =(0.9 ,0.999 ),eps =1e-08 ,weight_decay =0 ,amsgrad =False )

    loss_func =nn .MSELoss () # Define the loss function
    dfhistory =pd .DataFrame (columns =["epoch","train_loss","val_loss","train_median_pcc","val_median_pcc"])
    print ("Start Training...")
    nowtime =datetime .datetime .now ().strftime ('%Y-%m-%d %H:%M:%S')
    log_step_freq =20   # Print results once every this many steps

    print ("=========="*8 +"%s"%nowtime )
    best_val_median_pcc =0 
    record_file =open (output_path +'/'+test_sample +'-'+'best_epoch.csv',mode ='w')
    record_file .write ("epoch,best_val_median_pcc\n")
    loss_draw_train =[]
    loss_draw_test =[]
    pcc_draw_train =[]
    pcc_draw_test =[]

    stop_id =0 
    for epoch in range (1 ,epoch ):
        my_model .train ()
        loss_train_sum =0.0 
        epoch_median_pcc_val =None 
        epoch_median_pcc_train =None 

        epoch_real_record_train =[]
        epoch_predict_record_train =[]
        step_train =0 
        for stepi ,(patch_name ,imgs ,orig_exp ,genes )in enumerate (train_loader ,1 ):  # 1 is the optional start index so stepi starts at 1 instead of 0
          # Print(stepi,end="") #each img corresponds to one spot,
            step_train =stepi 
            optimizer .zero_grad ()
            if use_gpu :
                imgs =imgs .cuda ()
                genes =genes .cuda ()
                orig_exp =orig_exp .cuda ()

            if config ['random']==True :  # If performing the noise_exp ablation
                orig_exp =torch .randn (orig_exp .shape ).cuda ()
            predictions =my_model (imgs ,orig_exp ) # The model outputs predictions directly
            loss =loss_func (predictions ,genes )
            # print(f'epoch:{epoch} step:{stepi}==============loss:{float(loss)}')
              # Backpropagation to compute gradients
            loss .backward ()  # Backpropagate to compute gradients for all parameters
            optimizer .step ()  # Update parameters using the optimizer

            if use_gpu :
                predictions =predictions .cpu ().detach ().numpy ()  # Move predictions from GPU to CPU for saving
            else :
                predictions =predictions .detach ().numpy ()
                 # When training with image inputs, the image order is usually shuffled, so it's uncertain how many images will appear in the first few steps or whether they come from the same section
            epoch_real_record_train +=list (genes .cpu ().numpy ())
            epoch_predict_record_train +=list (predictions )
            epoch_median_pcc_train =compare_prediction_label_list (epoch_predict_record_train ,epoch_real_record_train ,config ['num_gene']) # Compute the median PCC for each step
            if use_gpu :
                loss_train_sum +=loss .cpu ().item () # Use .item() to get a Python number
            else :
                loss_train_sum +=loss .item ()

            gc .collect () # Used to manually trigger garbage collection
            if stepi %log_step_freq ==0 : # Print results every N batches #Log once every 20 steps?
                print (("training: [epoch = %d, step = %d, images = %d] loss: %.3f, "+"median pearson coefficient"+": %.3f")%
                (epoch ,stepi ,stepi *batch_size ,loss_train_sum /stepi ,epoch_median_pcc_train ))


        my_model .eval () # Evaluate model performance on the test samples each epoch
        loss_val_sum =0.0 
        epoch_real_record_val =[]
        epoch_predict_record_val =[]
        step_val =0 
        for stepi ,(patch_name ,imgs ,orig_exp ,genes )in enumerate (test_loader ,1 ):
        #print(stepi, end="")
            step_val =stepi 
            with torch .no_grad ():
                if use_gpu :
                    imgs =imgs .cuda ()
                    genes =genes .cuda ()
                    orig_exp =orig_exp .cuda ()
                if config ['random']==True :
                    orig_exp =torch .randn (orig_exp .shape ).cuda ()
                predictions =my_model (imgs ,orig_exp ) # The model outputs predictions directly
                loss =loss_func (predictions ,genes )

                if use_gpu :
                    loss_val_sum +=loss .cpu ().item () # Use .item() to get a Python number
                else :
                    loss_val_sum +=loss .item ()

                if use_gpu :
                    predictions =predictions .cpu ().detach ().numpy ()
                else :
                    predictions =predictions .detach ().numpy ()

            epoch_real_record_val +=list (genes .cpu ().numpy ())
            epoch_predict_record_val +=list (predictions )
            epoch_median_pcc_val =compare_prediction_label_list (epoch_predict_record_val ,epoch_real_record_val ,config ['num_gene'])

            if stepi *2 %log_step_freq ==0 : # Print results every N batches
                print ("validation sample",test_sample )
                print (("validation: [step = %d] loss: %.3f, "+"median pearson coefficient"+": %.3f")%
                (stepi ,loss_val_sum /stepi ,epoch_median_pcc_val ))

        historyi =(
        epoch ,loss_train_sum /step_train ,loss_val_sum /step_val ,epoch_median_pcc_train ,epoch_median_pcc_val )

        dfhistory .loc [epoch -1 ]=historyi 


        loss_draw_train .append (loss_train_sum /step_train ) # Plot the training and validation curves
        loss_draw_test .append (loss_val_sum /step_val )
        pcc_draw_train .append (epoch_median_pcc_train )
        pcc_draw_test .append (epoch_median_pcc_val )

        print (model_name )
        print ((
        "\nEPOCH = %d, loss_train_avg = %.3f, loss_val_avg = %.3f, epoch_median_pcc_train = %.3f, epoch_median_pcc_val = %.3f")
        %historyi )

        nowtime =datetime .datetime .now ().strftime ('%Y-%m-%d %H:%M:%S')
        print ("\n"+"=========="*8 +"%s"%nowtime )
        if epoch >=1 :
            if epoch_median_pcc_val >best_val_median_pcc : # If validation performance improves, save the latest model weights
                stop_id =0 
                best_val_median_pcc =epoch_median_pcc_val 
                print ("Sample:",test_sample ,"best epoch now:",epoch )
                record_file .write (str (epoch )+","+str (epoch_median_pcc_val )+"\n")
                record_file .flush () # Equivalent to flushing before the file is closed
                torch .save (my_model .state_dict (), # Save every model for which the validation PCC increases
                output_path +"/"+test_sample +"-"+"SciSt-best.pth")
            else :
                stop_id +=1 
                 # If stop_id==10: #Optionally enable an early stopping strategy
                #     break
    draw_loss (len (loss_draw_train ),loss_draw_train ,len (loss_draw_test ),loss_draw_test ,output_path )
    draw_pcc (len (pcc_draw_train ),pcc_draw_train ,len (pcc_draw_test ),pcc_draw_test ,output_path )
    record_file .close ()
    dfhistory .to_csv (output_path +'/'+test_sample +'_train_record.csv',index =False )

if __name__ =='__main__':
    warnings .filterwarnings ('ignore')
    parser =argparse .ArgumentParser ()
    parser .add_argument ('--cfg',type =str ,help ='hyperparameters path')
    args =parser .parse_args ()
    with open (args .cfg ,'r')as f :
        config =yaml .safe_load (f )
    scist_train (config )