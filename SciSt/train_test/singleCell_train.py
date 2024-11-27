import os
import numpy as np
import sys
sys.path.append('/2data/liyixin/HE2ST/03Code')
import pandas as pd
import torch
import torch.nn as nn
import yaml
import argparse
import datetime
import gc
import random
import warnings
from torch.utils.data import DataLoader
from SingleCell_Ref.dataset_utils.my_dataset import load_dataset
from paperTCGN.TCGN_model.TCGN import TCGN
from SingleCell_Ref.utils.draw_loss_pcc import draw_loss,draw_pcc
from SingleCell_Ref.singleCell_models.sc_guide_model import get_model


use_gpu = torch.cuda.is_available() # gpu加速
#use_gpu=False
torch.cuda.empty_cache() # 清除显卡缓存

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

def ST_SC_guide(config):  #test sample number指的是fold
# def ST_TCGN(test_sample_number, model_name, output_path, wts_path=None):  # test sample number指的是fold
    batch_size=config['batch_size']
    epoch=config['epoch']
    starttime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========" * 8 + "%s" % starttime)
    print("GPU available:", use_gpu)
    # load data
    train_loader,test_loader,test_sample = load_dataset(config)
    print("finish loading")
    print(test_sample)

    # initialize model
    import os
    # model_name = "TCGN_Orig"
    model_name = config['model_name']
    output_path = os.path.join(config['output_path'],test_sample)
    if not os.path.exists(output_path):
        os.makedirs(output_path)   #用来保存训练好的模型
    if model_name!='Sc_Guide_Model_with_CNN_SASM' and model_name!='Concat_Model_for_Ablation':
        my_model = get_model(config['model_name'])()
    else:
        my_model = get_model(config['model_name'])(config['img_encoder'],config['num_gene'])

    if use_gpu:
        my_model = my_model.cuda()
    if config['wts_path']!=None:
        my_model.load_state_dict(torch.load(config['wts_path']), strict=False)  #这个应该是timm包里的预训练模型

    # train the model
    optimizer = torch.optim.Adam(my_model.parameters(),lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    loss_func = nn.MSELoss()   #直接用MSE，这里在自己的方法中需要优化
    dfhistory = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "train_median_pcc", "val_median_pcc"])  #在计算pcc的时候，需要把属于同一患者的patch拼凑起来，因为是计算所有spots对于同一个基因的pcc，这是不是会有些问题
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_step_freq = 20

    print("==========" * 8 + "%s" % nowtime)
    from paperTCGN.TCGN_utils.real_metric import compare_prediction_label_list
    best_val_median_pcc = 0
    record_file = open(output_path + '/'+test_sample+'-'+'best_epoch.csv', mode='w')
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
        for stepi, (patch_name, imgs, orig_exp, genes) in enumerate(train_loader, 1):  #1代表可选的起始索引值，代表stepi从1开始，而不是0
            #print(stepi,end="")   #每个img是一个spot，
            step_train = stepi
            optimizer.zero_grad()
            if use_gpu:
                imgs = imgs.cuda()
                genes = genes.cuda()
                orig_exp = orig_exp.cuda()
            if config['model_output_num']==1:
                if config['random'] == True:
                    orig_exp = torch.randn(orig_exp.shape).cuda()
                predictions = my_model(imgs,orig_exp)  #模型直接输出predictions
                loss = loss_func(predictions, genes)
            else:
                tcgn_output,predictions = my_model(imgs,orig_exp)
                loss_tcgn = loss_func(tcgn_output,genes)
                loss_orig = loss_func(predictions,genes)
                loss = (1-config['loss_coefficient']) * loss_tcgn+ config['loss_coefficient'] * loss_orig
              #目前来看mseloss，似乎是用spot的所有基因来求的
            # print(f'epoch:{epoch} step:{stepi}==============loss:{float(loss)}')
            ## 反向传播求梯度
            loss.backward()  # 反向传播求各参数梯度
            optimizer.step()  # 用optimizer更新各参数

            if use_gpu:
                predictions = predictions.cpu().detach().numpy()
            else:
                predictions = predictions.detach().numpy()
            #在输入img进行训练的时候，img的顺序一般是乱的，因此不能确定前几个step会输入多少img以及是否来自同一个section
            epoch_real_record_train += list(genes.cpu().numpy())
            epoch_predict_record_train += list(predictions)
            epoch_median_pcc_train = compare_prediction_label_list(epoch_predict_record_train, epoch_real_record_train,config['num_gene'])#为了减少计算量，这句话应该在跳出step循环后写，往前一个tab
            #上面这个算得是所有spot（不分img）对于某个基因的pcc——有意义吗？
            if use_gpu:
                loss_train_sum += loss.cpu().item()  # 返回数值要加.item
            else:
                loss_train_sum += loss.item()

            gc.collect() #是用于手动触发垃圾回收机制
            if stepi % log_step_freq == 0:  # 当多少个batch后打印结果  #每20个step打印一次结果？
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
                if config['model_output_num'] == 1:
                    if config['random']==True:
                        orig_exp=torch.randn(orig_exp.shape).cuda()
                    predictions = my_model(imgs, orig_exp)  # 模型直接输出predictions
                    loss = loss_func(predictions, genes)
                else:
                    tcgn_output, predictions = my_model(imgs, orig_exp)
                    loss_tcgn = loss_func(tcgn_output, genes)
                    loss_orig = loss_func(predictions, genes)
                    loss = (1-config['loss_coefficient']) * loss_tcgn + config['loss_coefficient'] * loss_orig


                if use_gpu:
                    loss_val_sum += loss.cpu().item()  # 返回数值要加.item
                else:
                    loss_val_sum += loss.item()

                if use_gpu:
                    predictions = predictions.cpu().detach().numpy()
                else:
                    predictions = predictions.detach().numpy()

            epoch_real_record_val += list(genes.cpu().numpy())
            epoch_predict_record_val += list(predictions)
            epoch_median_pcc_val = compare_prediction_label_list(epoch_predict_record_val, epoch_real_record_val,config['num_gene'])

            if stepi * 2 % log_step_freq == 0:  # 当多少个batch后打印结果
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
                record_file.flush()  #相当于是在文件关闭前刷新它
                torch.save(my_model.state_dict(),   #对于每一个val pcc升高的模型都保存下来
                           output_path + "/"  + test_sample+"-"+ "ST_Net-" + model_name + "-best.pth")
            else:
                stop_id+=1
            # if stop_id==10:
            #     break
    draw_loss(len(loss_draw_train),loss_draw_train,len(loss_draw_test),loss_draw_test,output_path)
    draw_pcc(len(pcc_draw_train), pcc_draw_train, len(pcc_draw_test), pcc_draw_test,output_path)
    record_file.close()
    dfhistory.to_csv(output_path+'/'+test_sample+'_train_record.csv',index=False)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='hyperparameters path')
    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
    ST_SC_guide(config)