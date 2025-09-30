**Installation**
示例，后面按照这种格式来
OS: Linux (Tested on Ubuntu 18.04) 

Configure [conda env](docs/ABRS-P.yml) and 

Install the modified [timm](https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view?usp=sharing) library
```bash
pip install timm-0.5.4.tar
```
示例，后面按照这种格式来


**# Installation**
```
bash
conda create -n scist python=3.9 -y
conda activate scist
pip install -r requirements.txt
```

```
import torch
import numpy as np
from models import SciSt

my_model=SciSt('resnet34',785)# predict 785 genes
# load pretrained model
my_model.load_state_dict(torch.load('./Weights/A2-TCGN-her2-best.pth'), strict=True)
# use gpu to run the model
my_model = my_model.cuda()
# imgs: input batched image tensor on the gpu with shape Batch_size x 3 x 224 x 224
# if the magnification is 20x then orginal image size should be 112x112, and 56x56 for 10x magnification, then resize them to 224x224
# if the input img has the same size as the spot in the spatial transcriptomics and is not 224 x 224, then resize the img to 224 x 224
# img should be 0-1 normalized, then normalized by IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD in the python package timm
predictions=my_model(imgs,noise_exp).detach().cpu()
# predictions: predicted gene expression tensor with shape Batch_size x 785
# where 785 is the number of genes the model is set to output

# Specific for HER2+ dataset (breast cancer)
# Predict the expression of genes that can be statistically significantly predicted by the model
selected_genes_number_tensor=torch.Tensor(np.load("data/genes_her2_that_we_think_can_be_predicted.npy")).bool() #在data里上传785基因list
predictions=predictions[:,selected_genes_number_tensor]
# to numpy form if needed
predictions=predictions.numpy()
```

# System
Pytorch>=2.4.0
Pip>=23

# Download datasets
看一下TCGN怎么写的，仿照他的

# Steps
在进到正式的step之前，查询在linux进到目标dir的command line

## Train steps
You need to have a series of spatial transcriptomics datasets and one corresponding single-cell transcriptomics dataset.
### 1. Identify predicted genes
We provide one example to identify liver cancer spatial transcriptomics data hvg genes. 
```
bash
python ./src/dataset/generate_hvg_genes.py --h5_path YOUR_H5_PATH --save_path YOUR_SAVE_PATH
```
The predicted genes need to intersect with gene sets in both spatial transcriptomics datasets and single-cell transcriptomics dataset.


### 2. Preprocess
#### 2.1 Cut patches and save labels, centers
According to center coords, extend 112. Finally, get 224 X 224 patches, and then save the normalized labels according to predicted genes. Also save centers for visualization.
```
bash
python ./src/dataset/create_patch_label.py
```
This step will save data in following three folds:
./01_patch
./02_label
./03_center
#### 2.2 Segment patches
We implemented this by Hover-Net. It classify nucleus into five categories: neoplastic, inflammatory, connective, non-neoplastic epithelial, and dead cells. You need to provie the patch patch, model weights download path, and save path. We provide this command line.

```
bash
conda activate hover_net_env
cd HOVER_NET_PATH
python run_infer.py \
--gpu='0' \ #change it if you need
--nr_types=6 \
--type_info_path=./type_info.json \
--batch_size=32 \ #change it if you need
--model_mode=fast \
--model_path=./pretrained_model/hovernet_fast_pannuke_type_tf2pytorch.tar \ #change to your model weights path
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=YOUR_PATCH_PATH \
--output_dir=YOU_SAVE_PATH \ #./04-seg-results
--mem_usage=0.1 #change it if you need
```
This whill produce three folds: json, mat, and overlay. We only use json files.    
You can segmente other classifications or use other Segmentation models.

#### 2.3 Generate single-cell reference for each cell types
Choose approprite cell types for each one of above cell classifications.  
```
bash
python ref_preprocess.py
```
This step will produce ./05-ref.csv file
Same single-cell reference if ok if you do the same disease type.


#### 2.4 Generate noise_exp for each patch
If you classify cells into five classifications as we provided. You can run the following code to get the noise_exp for each patch.
```
bash
python create_orig_exp.py 
```
This step will produce ./06-noise_exp/ file


By now, the preprocess steps are over. We get 6 files totally.
1. patch files
2. label files
3. center files
4. seg_results files
5. ref files
6. noise_exp files


### 3. Training
#### 3.1 Define your config file
We provide a template of config file.

#### 3.2 Train
```
python ./src/scripts/main_scist_train.py --cfg CONFIG_PATH
```

### 4. Evaluate
```
bash
python evaluation.py
```

### 5. Visualize
Refer to jupyter_notebooks/visualization.ipynb


## Inference steps
You can do this when you have high resolution H&E images.
### 1. Preprocess
#### 1.1 Cut patches
Same as training step, except for the label saving.
#### 1.2 Segmente patches
Same as training step.
#### 1.3 Generate noise_exp files
Same as training step.
### 2. Inference
#### 2.1 Generate prediction file
```
bash
python inference.py
```