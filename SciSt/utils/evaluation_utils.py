import numpy as np
import pandas as pd
def calculate_pcc(x,y):
    n=len(x)
    sum_xy = np.sum(np.sum(x * y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x * x))
    sum_y2 = np.sum(np.sum(y * y))
    pc = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    return pc

def get_pccs(prediction_list,real_list):
    header = list(range(prediction_list[0].shape[0]))
    prediction_df = pd.DataFrame(columns=header, data=prediction_list)
    real_df = pd.DataFrame(columns=header, data=real_list)
    header = list(prediction_df.columns)
    pccs = []
    for genei in header:
        predictioni = prediction_df.loc[:, genei]
        reali = real_df.loc[:, genei]
        pcci = calculate_pcc(predictioni, reali)
        pccs.append(pcci)
    pccs = np.array(pccs)
    return pccs