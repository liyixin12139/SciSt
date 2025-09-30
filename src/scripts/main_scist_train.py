import torch 
import random 
import warnings 
import os 
import numpy as np 
from SingleCell_Ref .train_test .singleCell_train import scist_train 
import argparse 
import yaml 

 # Fix random seed for reproducibility
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
use_gpu =torch .cuda .is_available () # GPU acceleration
#use_gpu=False
torch .cuda .empty_cache () # Clear CUDA cache

warnings .filterwarnings ('ignore')
parser =argparse .ArgumentParser ()
parser .add_argument ('--cfg',type =str ,help ='hyperparameters path') # Read the cfg file passed in from bash
args =parser .parse_args ()
with open (args .cfg ,'r')as f :
    config =yaml .safe_load (f )

scist_train (config ) # Train SciSt on the test samples specified in the config