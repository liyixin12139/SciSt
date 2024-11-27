import numpy as np
import os
import matplotlib.pyplot as plt

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def draw_loss(epoch_train:int,loss_train:list,epoch_test:int,loss_test,save_path:str):

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.plot(range(1,epoch_train+1), loss_train, color='blue', linestyle="solid", label="train loss")
    plt.plot(range(1,epoch_test+1), loss_test, color='red', linestyle="solid", label="val loss")
    plt.legend()

    plt.title('Loss curve')

    save_path = os.path.join(save_path,'training_process')
    make_dir(save_path)
    plt.savefig(os.path.join(save_path,"loss.png"))


def draw_pcc(epoch_train:int,loss_train:list,epoch_test:int,loss_test,save_path:str):

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('Epoch')
    plt.ylabel('PCC')

    # plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.plot(range(1,epoch_train+1), loss_train, color='blue', linestyle="solid", label="train pcc")
    plt.plot(range(1,epoch_test+1), loss_test, color='red', linestyle="solid", label="val pcc")
    plt.legend()

    plt.title('PCC curve')

    save_path = os.path.join(save_path,'training_process')
    make_dir(save_path)
    plt.savefig(os.path.join(save_path,"pcc.png"))


if __name__ == '__main__':
    path = './02-combine_with_TCGN/02-add_embedding_layer_based_TCGN/test'
    epoch_train = 5
    epoch_test = 5
    loss_train = list(range(5))
    loss_test = list(range(3,8))
    print(loss_test)
    draw_pcc(int(epoch_train),loss_train,int(epoch_test),loss_test,path)

