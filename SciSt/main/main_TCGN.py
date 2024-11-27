
import torch
import random
import warnings
import os
import numpy as np
from SingleCell_Ref.train_test.singleCell_train import ST_TCGN
import os
import pandas as pd
import torch
import torch.nn as nn
import datetime
import gc
import random
import warnings
from SingleCell_Ref.singleCell_models.TCGN_based import TCGN_Orig,TCGN_Orig_Control
from torch.utils.data import DataLoader
from SingleCell_Ref.dataset_utils.my_dataset import HER2_dataset
from paperTCGN.TCGN_model.TCGN import TCGN
from SingleCell_Ref.utils.draw_loss_pcc import draw_loss,draw_pcc
from SingleCell_Ref.singleCell_models.sc_guide_model import Sc_Guide_Model

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def ST_TCGN_control(test_sample_number,model_name,output_path,wts_path = None):  #test sample number指的是fold
    batch_size=32
    epoch=31
    print("GPU available:", use_gpu)
    # load data
    patch_path = './01-data/01-her2数据集/01-gen-patch/'
    exp_label_path = './01-data/01-her2数据集/02-gene-exp-label/'
    ref_path = './01-data/01-her2数据集/04-singlecell_ref/filter_her2_ref.csv'
    predicted_gene_path = './01-HER2+/her2st-master/data/her_hvg_cut_1000.npy'
    seg_path = './01-data/01-her2数据集/03-patch-segResult/json/'
    orig_path = './01-data/01-her2数据集/05-orig_exp/'

    train_dataset = HER2_dataset(patch_path,exp_label_path,predicted_gene_path,ref_path,orig_path,seg_path,'train',test_sample_number)
    test_dataset = HER2_dataset(patch_path,exp_label_path,predicted_gene_path,ref_path,orig_path,seg_path,'test',test_sample_number)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    print("finish loading") 

    # initialize model
    import os
    dirs = output_path + model_name
    if not os.path.exists(dirs):
        os.makedirs(dirs)   
    if model_name=='TCGN_Orig':
        my_model = TCGN_Orig()
    elif model_name=='TCGN_Orig_control':
        my_model = TCGN_Orig_Control()
    elif model_name=='sc_guide_model':
        my_model = Sc_Guide_Model()
    elif model_name=='TCGN':
        my_model = TCGN()
    else:
        print('model name is invalid.')

    if use_gpu:
        my_model = my_model.cuda()
        if wts_path!=None:
            my_model.load_state_dict(torch.load(wts_path), strict=False)  


    optimizer = torch.optim.Adam(my_model.parameters(),lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    loss_func = nn.MSELoss()   
    dfhistory = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "train_median_pcc", "val_median_pcc"])  
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_step_freq = 20

    print("==========" * 8 + "%s" % nowtime)
    from paperTCGN.TCGN_utils.real_metric import compare_prediction_label_list
    best_val_median_pcc = 0
    test_sample = test_dataset.test_sample
    record_file = open(output_path + model_name + '/'+test_sample+'-'+'best_epoch.csv', mode='w')
    record_file.write("epoch,best_val_median_pcc\n")
    loss_draw_train = []
    loss_draw_test = []
    pcc_draw_train = []
    pcc_draw_test = []

    stop_id = 0
    for epoch in range(1, epoch):
        my_model.train()
        loss_train_sum = 0.0
        epoch_median_pcc_val = None
        epoch_median_pcc_train = None

        epoch_real_record_train = []
        epoch_predict_record_train = []
        step_train = 0
        for stepi, (patch_name, imgs, orig_exp, genes) in enumerate(train_loader, 1): 
            step_train = stepi
            optimizer.zero_grad()
            if use_gpu:
                imgs = imgs.cuda()
                genes = genes.cuda()
                orig_exp = orig_exp.cuda()
            predictions,_ = my_model(imgs) 
            loss = loss_func(predictions, genes)
            loss.backward()  
            optimizer.step()  

            if use_gpu:
                predictions = predictions.cpu().detach().numpy()
            else:
                predictions = predictions.detach().numpy()
            epoch_real_record_train += list(genes.cpu().numpy())
            epoch_predict_record_train += list(predictions)
            epoch_median_pcc_train = compare_prediction_label_list(epoch_predict_record_train, epoch_real_record_train)

            if use_gpu:
                loss_train_sum += loss.cpu().item()  
            else:
                loss_train_sum += loss.item()

            gc.collect()
            if stepi % log_step_freq == 0:  
                print(("training: [epoch = %d, step = %d, images = %d] loss: %.3f, " + "median pearson coefficient" + ": %.3f") %
                      (epoch, stepi, stepi*batch_size,loss_train_sum / stepi, epoch_median_pcc_train))


        my_model.eval()
        loss_val_sum = 0.0
        epoch_real_record_val = []
        epoch_predict_record_val = []
        step_val = 0
        for stepi, (patch_name, imgs, orig_exp, genes) in enumerate(test_loader, 1):
            #print(stepi, end="")
            step_val = stepi
            with torch.no_grad():
                if use_gpu:
                    imgs = imgs.cuda()
                    genes = genes.cuda()
                    orig_exp = orig_exp.cuda()
                predictions,_ = my_model(imgs)
                loss = loss_func(predictions, genes)

                if use_gpu:
                    loss_val_sum += loss.cpu().item()  
                else:
                    loss_val_sum += loss.item()

                if use_gpu:
                    predictions = predictions.cpu().detach().numpy()
                else:
                    predictions = predictions.detach().numpy()

            epoch_real_record_val += list(genes.cpu().numpy())
            epoch_predict_record_val += list(predictions)
            epoch_median_pcc_val = compare_prediction_label_list(epoch_predict_record_val, epoch_real_record_val)

            if stepi * 2 % log_step_freq == 0:  
                print("validation sample", test_sample)
                print(("validation: [step = %d] loss: %.3f, " + "median pearson coefficient" + ": %.3f") %
                      (stepi, loss_val_sum / stepi, epoch_median_pcc_val))

        historyi = (
            epoch, loss_train_sum / step_train, loss_val_sum / step_val, epoch_median_pcc_train, epoch_median_pcc_val)

        dfhistory.loc[epoch - 1] = historyi


        loss_draw_train.append(loss_train_sum/step_train)
        loss_draw_test.append(loss_val_sum/step_val)
        pcc_draw_train.append(epoch_median_pcc_train)
        pcc_draw_test.append(epoch_median_pcc_val)

        print(model_name)
        print((
                  "\nEPOCH = %d, loss_train_avg = %.3f, loss_val_avg = %.3f, epoch_median_pcc_train = %.3f, epoch_median_pcc_val = %.3f")
              % historyi)

        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % nowtime)
        if epoch >= 1:
            if epoch_median_pcc_val > best_val_median_pcc:
                stop_id=0
                best_val_median_pcc = epoch_median_pcc_val
                print("Sample:",test_sample,"best epoch now:", epoch)
                record_file.write(str(epoch) + "," + str(epoch_median_pcc_val) + "\n")
                record_file.flush() 
                torch.save(my_model.state_dict(),   
                           output_path + model_name + "/"  + test_sample+"-"+ "ST_Net-" + model_name + "-best.pth")
            else:
                stop_id+=1
            # if stop_id==10:
            #     break
    save_path = output_path+model_name
    draw_loss(len(loss_draw_train),loss_draw_train,len(loss_draw_test),loss_draw_test,save_path)
    draw_pcc(len(pcc_draw_train), pcc_draw_train, len(pcc_draw_test), pcc_draw_test,save_path)
    record_file.close()
    dfhistory.to_csv(output_path+model_name+'/'+test_sample+'_train_record.csv',index=False)
seed_torch()
use_gpu = torch.cuda.is_available()  # gpu加速
# use_gpu=False
torch.cuda.empty_cache() 

warnings.filterwarnings('ignore')
output_path = './01-TCGN-used-for-cntrast/224/continue_train/'
wts_path = './01-TCGN-used-for-cntrast/224/TCGN/A2-ST_Net-TCGN-best.pth'
ST_TCGN_control(0, 'TCGN',output_path,wts_path)