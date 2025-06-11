import os
def find_her_index(index):
    samples_dict={0:'A2',1:'A3',2:'A4',3:'A5',4:'A6',5:'B1',6:'B2',7:'B3',8:'B4',
    9:'B5',10:'B6',11:'C1',12:'C2',13:'C3',14:'C4',15:'C5',16:'C6',17:'D1',18:'D2',
    19:'D3',20:'D4',21:'D5',22:'D6',23:'E1',24:'E2',25:'E3',26:'F1',27:'F2',28:'F3',
    29:'G1',30:'G2',31:'G3'}
    if type(index)==str:
        for k,v in samples_dict.items():
            if v==index:
                return k
    else:
        for k,v in samples_dict.items():
            if k==index:
                return v
def find_cscc_index(index):
    samples_dict={0:'P10_ST_rep1',1:'P10_ST_rep2',2:'P10_ST_rep3',3:'P2_ST_rep1',4:'P2_ST_rep2',5:'P2_ST_rep3',6:'P5_ST_rep1',7:'P5_ST_rep2',8:'P5_ST_rep3',
    9:'P9_ST_rep1',10:'P9_ST_rep2',11:'P9_ST_rep3'}
    if type(index)==str:
        for k,v in samples_dict.items():
            if v==index:
                return k
    else:
        for k,v in samples_dict.items():
            if k==index:
                return v
if __name__ == '__main__':
    index=8
    print(find_her_index(index))