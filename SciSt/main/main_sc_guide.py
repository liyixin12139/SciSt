import torch
import random
import warnings
import os
import numpy as np
from SingleCell_Ref.train_test.singleCell_train import ST_SC_guide
import argparse
import yaml

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()
use_gpu = torch.cuda.is_available() # gpu加速
#use_gpu=False
torch.cuda.empty_cache() # 清除显卡缓存

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--cfg',type=str,help='hyperparameters path')
args = parser.parse_args()
with open(args.cfg,'r') as f:
    config = yaml.safe_load(f)

ST_SC_guide(config)