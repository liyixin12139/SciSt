
# SciSt: Spatial Transcriptomics Gene Expression Prediction

> Using deep learning to predict gene expression from tissue slide images (`SciSt`).

This repository provides inference examples and environment setup for loading the pre-trained `SciSt` model and performing gene expression prediction on batches of images using `PyTorch`. This document aims to be **formal, clear, and reproducible**, with detailed instructions on input/output, preprocessing.

---

## Table of Contents
- [Overview](#overview)
- [Environment Requirements](#environment-requirements)
- [Installation and Setup](#installation-and-setup)
- [Quick Start (Inference)](#quick-start-inference)
- [Project Structure (Example)](#project-structure-example)
- [Download Datasets](#download-datasets)
- [Steps](#Steps)
- [License and Acknowledgements](#license-and-acknowledgements)

---

## Overview
- Load `SciSt` model (Example: `resnet34` backbone, outputting 785 genes).
- Load pre-trained weights from `./Weights/A2-scist-best.pth` and perform inference.
- Clear guidelines for training and inference steps.

---

## Environment Requirements
- OS: Linux (Tested on Ubuntu 20.04)
- Python: 3.9 (Conda environment recommended)
- Dependencies: See `requirements.txt` (includes `torch`, `torchvision`, `timm`, etc.)

> **Note**: Specific framework and library versions will be consistent with `requirements.txt`.

---

## Installation and Setup

### 1) Create and Activate Conda Environment
```bash
conda create -n scist python=3.9 -y
conda activate scist
```

### 2) Install Dependencies
```bash
pip install -r requirements.txt
```

### 3) Prepare Weights
Place the pre-trained weight file in:
```
./Weights/A2-scist-best.pth
```

---

## Quick Start (Inference)

The following example shows how to load the model and perform inference on a batch of images. Replace `load_your_batch()` with your actual data loading implementation.

```python
import torch
from models import SciSt

# ------------------------------
# 1. Device and Random Seed
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ------------------------------
# 2. Initialize Model
#    Using resnet34 backbone, predicting 785 genes
# ------------------------------
model = SciSt('resnet34', 785)

# ------------------------------
# 3. Load Pre-trained Weights
# ------------------------------
state = torch.load('./Weights/A2-scist-best.pth', map_location='cpu')
model.load_state_dict(state, strict=True)
model.to(device)
model.eval()  # Set to evaluation mode before inference

# ------------------------------
# 4. Prepare Input
# imgs: [B, 3, 224, 224] Tensor on GPU
# noise_exps: [B, num_predicted_genes]
# ------------------------------


# ------------------------------
# 5. Predict
# ------------------------------
predictions = model(imgs,noise_exps) 
# predictions: [B, 785] — Each sample's gene expression prediction vector
predictions_cpu = predictions.detach().cpu()
print(predictions_cpu.shape)
```

---

## Project Structure

> This is an example of the key file locations. Actual structure may vary.

```csharp
.
├─ Weights             # SciSt's weights for specific samples  
├─ configs             # Configuration templates for training SciSt  
├─ Jupyter_notebooks/  # Jupyter notebooks for analysis and visualization 
│   ├─ data         
│   └─visualization.ipynb
├─ src/                # Source code for the project
│   ├─ dataset         # Dataset handling and preprocessing scripts  
│   ├─ models          # Model definitions (SciSt model and related) 
│   ├─ scripts         # Scripts for running the model
│   ├─ train_test      # Scripts for training and testing the model
│   └─ utils           # Utility functions
└─ README.md
```

---

## Download Datasets
- **Human HER2-positive breast tumor ST data**: [GitHub - HER2ST](https://github.com/almaan/her2st/)
- **Human cutaneous squamous cell carcinoma 10x Visium data**: [GSE144240 (NCBI GEO)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144240)

---

## Steps

### Train Steps
To train the model, you will need a series of spatial transcriptomics datasets and a corresponding single-cell transcriptomics dataset.

#### 1. Identifying Predicted Genes
We provide an example of identifying **HVG genes** for liver cancer spatial transcriptomics data. You can run the following command to identify the predicted genes:
```bash
python ./src/dataset/generate_hvg_genes.py --h5_path YOUR_H5_PATH --save_path YOUR_SAVE_PATH
```
The predicted genes should be cross-referenced with gene sets from both the spatial transcriptomics datasets and the single-cell transcriptomics dataset.

#### 2. Preprocessing

##### 2.1 Cutting Patches and Saving Labels, Centers
Based on the center coordinates, extend by 112 units, and obtain 224 × 224 patches. These patches should be saved along with the corresponding normalized labels based on the predicted genes. Additionally, save the centers for later visualization. Run the following command to perform this step:
```bash
python ./src/dataset/create_patch_label.py
```

This process will generate and save the following three sets of data:

1. Patch files
2. Label files
3. Center files

##### 2.2 Patch Segmentation
Segmentation of patches is performed using **Hover-Net**, which classifies nuclei into five distinct categories: 
- Neoplastic
- Inflammatory
- Connective
- Non-neoplastic epithelial
- Dead cells

To run the segmentation, provide the patch directory, model weights path, and the output directory. The following command can be used:

```bash
conda activate hover_net_env
cd HOVER_NET_PATH
python run_infer.py \
--gpu='0' \ 
--nr_types=6 \
--type_info_path=./type_info.json \
--batch_size=32 \  
--model_mode=fast \
--model_path=./pretrained_model/hovernet_fast_pannuke_type_tf2pytorch.tar \ 
--nr_inference_workers=8 \
--nr_post_proc_workers=16 tile \
--input_dir=YOUR_PATCH_PATH \
--output_dir=YOU_SAVE_PATH \  # ./04-seg-results
--mem_usage=0.1 
```
This will generate three output folders: **json**, **mat**, and **overlay**. Only the json files are needed for subsequent steps.
You may also choose to segment using other models or classifications.

##### 2.3 Generate Single-cell Reference for Each Cell Type
For each of the classifications mentioned above, choose appropriate cell types and generate the reference file by running:
```bash
python ref_preprocess.py
```
This command will create the `./05-ref.csv` file.  
The same single-cell reference file can be used if the disease type is consistent across datasets.

##### 2.4 Generating `noise_exp` for Each Patch
If you have classified cells into the five categories as described, use the following command to generate the `noise_exp` data for each patch:
```bash
python create_orig_exp.py
```
This will produce the `./06-noise_exp/` directory.

After completing the preprocessing steps, you should have the following six directories and files:

```csharp
.
├─ ./01_patch/    
├─ ./02_label/   
├─ ./03_center/
├─ ./04-seg-results/
├─ ./05-ref.csv/       
└─ ./06-noise_exp/

```

---

### 3. Training

#### 3.1 Config File Setup
We provide a template configuration file that should be adjusted to match your specific dataset and training requirements.

#### 3.2 Begin Training
To start training, run the following command, making sure to specify the path to your configuration file:
```bash
python ./src/scripts/main_scist_train.py --cfg CONFIG_PATH
```

---

### 4. Model Evaluation
Once the model has been trained, you can evaluate its performance using the following command:
```bash
python evaluation.py
```

---

### 5. Visualizing Results
For visualizing the model's results, refer to the Jupyter notebook located at `jupyter_notebooks/visualization.ipynb`.

---

## Inference Steps
The following steps can be used when you have high-resolution H&E images and wish to perform inference.

### 1. Preprocessing

#### 1.1 Cutting Patches
The patch cutting process is the same as the training step, but without the need to save labels.

#### 1.2 Segmenting Patches
This step is identical to the one performed during training.

#### 1.3 Generating `noise_exp` Files
Follow the same procedure as in the training steps to generate the `noise_exp` files.

### 2. Running Inference
Once the preprocessing steps are complete, you can generate predictions from the trained model using the following command:
```bash
python inference.py
```
---

## License and Acknowledgements
- **License**: `TODO` (Please choose and fill in, e.g., MIT/Apache-2.0, etc.)
- **Acknowledgements**: Thanks to community open-source tools (e.g., `PyTorch`, `torchvision`, `timm`) for their support.

---

**Maintainers**: _(Please provide your name and contact information)_  
**Issue Reporting**: Feel free to report issues or submit PRs via GitHub.
