import os 
import torch 
import numpy as np 
import pandas as pd 
from models .sc_guide_model import SciSt 
from torch .utils .data import DataLoader 
from statsmodels .stats import multitest 
from utils .fine_number import find_her_index ,find_cscc_index 


class Initiate_dataset (Dataset ):
    def __init__ (self ,patch_path ,orig_path ,mode ,fold ,data_name ):
        super (Initiate_dataset ,self ).__init__ ()
        self .patch_path =patch_path 
        self .orig_path =orig_path 
        self .mode =mode 
        self .wsi_ids_lst =list (set ([i .split ('_')[0 ]for i in os .listdir (patch_path )]))
        self .wsi_ids_lst .sort ()
        self .train_transform =transforms .Compose (
        [transforms .RandomRotation (180 ),
        transforms .RandomHorizontalFlip (0.5 ),
        transforms .RandomVerticalFlip (0.5 ),
        ]
        )
        from timm .data .constants import IMAGENET_DEFAULT_MEAN ,IMAGENET_DEFAULT_STD 
        self .basic_transform =transforms .Compose (
        [transforms .Resize ((224 ,224 ),antialias =True ),# resize to 256x256 square
        transforms .ConvertImageDtype (torch .float ),
        transforms .Normalize (IMAGENET_DEFAULT_MEAN ,IMAGENET_DEFAULT_STD )
        ]
        )
        print ('Loading img...')
        if mode !='train':
            self .wsi_ids =[self .wsi_ids_lst [fold ]]
            self .test_sample =self .wsi_ids [0 ]

        else :
            self .wsi_ids =list (set (self .wsi_ids_lst )-set ([self .wsi_ids_lst [fold ]]))#31
        if data_name =='her2':
            self .patch_meta ={wsi_i :[j for j in os .listdir (self .patch_path )if j .split ('_')[0 ]==wsi_i ]for wsi_i in self .wsi_ids }
            self .patch_ids =[]
            for wsi_i in list (self .patch_meta .keys ()):
                self .patch_ids .extend (self .patch_meta [wsi_i ])

        elif data_name =='cscc':
            self .patch_meta ={wsi_i :[j for j in os .listdir (self .patch_path )if '_'.join (j .split ('_')[:-1 ])==wsi_i ]for wsi_i in 
            self .wsi_ids }
            self .patch_ids =[]
            for wsi_i in list (self .patch_meta .keys ()):
                self .patch_ids .extend (self .patch_meta [wsi_i ])

    def __getitem__ (self ,item ):

        patch_name =self .patch_ids [item ]
        patch =self .get_img (patch_name )#(3,224,224)
        patch_orig_path =os .path .join (self .orig_path ,patch_name .replace ('png','npy'))
        orig_exp =torch .Tensor (np .load (patch_orig_path )).squeeze ()
        data =(patch_name ,patch ,orig_exp )
        return data 


    def __len__ (self ):
        return len (self .patch_ids )

    def get_img (self ,img_name ):
        img_path =os .path .join (self .patch_path ,img_name )
        img =Image .open (img_path )
        img =torch .Tensor (np .array (img )/255 )
        img =img .permute (1 ,0 ,2 )
        img =img .permute (2 ,0 ,1 )
        return self .transforms (img )

    def transforms (self ,img ):
        my_transforms =[self .basic_transform ,self .train_transform ]
        if self .mode !='train':
            return my_transforms [0 ](img )
        else :
            return my_transforms [1 ](my_transforms [0 ](img ))

def generate_label_prediction_center (patch_path ,orig_path ,model_result_path ,samples =None ):
    model =Sc_Guide_Model_with_CNNSASM ('resnet34',785 ) # Initialize the model
    if samples ==None : # No extra inference sample specified
        samples =os .listdir (model_result_path )
        samples .sort ()
    for fold in range (len (samples )):
        sample =samples [fold ]
        test_index =find_her_index (sample )
        dataset =Initiate_dataset (patch_path ,orig_path ,'test',test_index ,'her2') # Load the test dataset

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
                patch_name ,imgs ,orig_exp =item [0 ],item [1 ],item [2 ]
                imgs ,orig_exp =imgs .to ('cuda:0'),orig_exp .to ('cuda:0')
                pred =model (imgs ,orig_exp ) # This is the input format for sc-guide
                if len (pred .shape )==1 : # Different models may output different shapes for evaluation, e.g., (1,785) vs (785,)
                    preds_list .append (pred .unsqueeze (0 ).cpu ().detach ().numpy ())
                    print ('pred_list shape',pred .shape )
                else :
                    preds_list .append (pred .cpu ().detach ().numpy ())


        preds_list =np .concatenate (preds_list ,axis =0 ) # Concatenate predictions for each patch and save them together

        save_path =os .path .join (model_result_path ,test_sample )
        np .save (os .path .join (save_path ,'preds.npy'),preds_list )


if __name__ =='__main__':

    patch_path ='./01_patch/'
    orig_path ='./06-noise_exp/'
    model_result_path ='./07-SciSt/'
     # Inference
    generate_label_prediction_center (patch_path ,orig_path ,model_result_path )
