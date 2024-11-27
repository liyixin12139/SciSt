from sklearn.metrics import roc_curve,auc
import os
import matplotlib.pyplot as plt
def plot_roc(true_lst,pred_lst,save_path):
    fpr, tpr, threshold = roc_curve(true_lst, pred_lst)
    roc_auc = auc(fpr, tpr)  # 准确率代表所有正确的占所有数据的比值
    print('roc_auc:', roc_auc)
    lw = 2
    plt.subplot(1, 1, 1)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1 - specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC', y=1)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path,'roc.png'))
    plt.close()
if __name__ == '__main__':
    a=[1,1,0,1,0]
    b=[0.9,0.4,0.8,0.3,0.2]
    save_path='./03-without_TCGN_as_ImgEncoder/03-CNN-SASM/HER_TLS_retrain_with_2xNumgene/'
    plot_roc(a,b,save_path)
