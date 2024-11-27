import json
import os
import numpy as np
from paperTCGN.TCGN_Dataset.dataset import pk_load
from torch.utils.data import DataLoader

#随后在这里将生成patch和label的代码整理为类


if __name__=="__main__":
    trainset = pk_load(0, 'train', False, 'her2st', neighs=4, prune='Grid')
    train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)   #batch size只能是1
    # for stepi, input in enumerate(train_loader, 1):
    #     for i in range(len(input)):
    #         '''
    #         img <class 'torch.Tensor'> torch.Size([1, 325, 3, 112, 112])
    #         position <class 'torch.Tensor'> torch.Size([1, 325, 2])
    #         exp <class 'torch.Tensor'> torch.Size([1, 325, 785])
    #         <class 'torch.Tensor'> torch.Size([1, 325, 325])
    #         <class 'torch.Tensor'> torch.Size([1, 325, 785])
    #         <class 'torch.Tensor'> torch.Size([1, 325])
    #         <class 'torch.Tensor'> torch.Size([1, 325, 2])
    #         '''
    #         print(type(input[i]),input[i].shape)

    #保存her2数据集的patch和基因表达label
    import cv2
    save_path = './01-orig gene exp/01-her2数据集/01-gen-patch/'
    gene_save_path = './01-orig gene exp/01-her2数据集/02-gene-exp-label/'
    for item in train_loader:
        _id = item[-1][0]
        imgs = item[0].squeeze(0)  #(n,3,224,224)
        exps = item[2].squeeze(0) #(n,785)
        exp_file = {}
        for i_img in range(imgs.shape[0]):
            img = np.array(imgs[i_img].permute(2,1,0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            save_name = _id+'_'+str(i_img)
            cv2.imwrite(os.path.join(save_path,save_name+'.png'),img)
            exp_file[save_name] = np.array(exps[i_img]).tolist()
        with open(os.path.join(gene_save_path,_id+'.json'),'w') as f:
            json.dump(exp_file,f)