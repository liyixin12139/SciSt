# Usage
```
pip install -r requirements.txt
```

# System
Pytorch>=2.4.0

# Train steps
You need to have a series of spatial transcriptomics datasets and one corresponding single-cell transcriptomics dataset.
## 1. Identify predicted genes
It has to be the intersection gene sets in both spatial transcriptomics datasets and single-cell transcriptomics dataset.
You can find the code in xxx.ipynb.

## 2. Preprocess
### 2.1 Cut patches and save labels
According to center coords, extend 112. Finally, get 224 X 224 patches, and then save the normalized labels according to predicted genes. Also save centers for visualization.
### 2.2 Segment patches
We implemented this by Hover-Net. It classify nucleus into five categories: neoplastic, inflammatory, connective, non-neoplastic epithelial, and dead cells. You need to provie the patch patch, model weights download path, and save path. We provide this command line.

```
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
--output_dir=YOU_SAVE_PATH \
--mem_usage=0.1 #change it if you need
```
This whill produce three folds: json, mat, and overlay. We only use json files.    
You can segmente other classifications or use other Segmentation models.
### 2.3 Generate noise_exp for each patch
If you classify cells into five classifications as we provided. You can run the following code to get the noise_exp for each patch.
```
python create_orig_exp.py #这个文件后面要修改
```
### 2.4 Generate single-cell reference for each cell types
Choose approprite cell types for each one of above cell classifications.  
Refer to xxx.py

By now, the preprocess steps are over. We get 5 files totally.
1. patch files
2. label files
3. center files
4. noise_exp files
5. reference files

## 3. Training
### 3.1 Define your config file
We provide a template of config file.

### 3.2 Train
```
python XXX.py --cfg CONFIG_PATH
```

## 4. Evaluation
### 4.1 Generate PCC file

### 4.2 Generate p_value file

## 5. Visualization
Refer to XXX.ipynb