import os
import json
import scprep as scp
import numpy as np
import pandas as pd


def count_celltype(json_path,id2tissue_dict)->dict:
    '''
    Collect counts of different cell types from HoverNet segmentation result JSON files
    :param json_path: .json file
    :param id2tissue_dict: dict
    '''
    with open(json_path,'r') as f:
        json_file = json.load(f)
    json_nuc_dict = json_file['nuc'] #get class labels

    count_dict = {}
    for celltype in list(id2tissue_dict.values()): #add cell types as keys to the dict
        count_dict[celltype] = 0

    for cell_i in json_nuc_dict.values():
        sin_celltype = id2tissue_dict[str(cell_i['type'])]
        count_dict[sin_celltype]+=1 #count each cell type in the patch
    f.close()
    return count_dict


def substitute_gene(gene_list,orig_gene,update_gene):
    gene_list_update = [update_gene if i==orig_gene else i for i in gene_list]
    return gene_list_update

def process_ref(ref_path)->pd.DataFrame:
    '''
    Process the ref file: set gene names as row index and transpose to shape (cell_num, gene_num)
    :param ref_path: csv file
    :return:
    '''
    ref = pd.read_csv(ref_path)
    ref.set_index('Unnamed: 0', inplace=True)
    return ref

def create_orig_exp(count_dict,sincell_ref):
    '''
    Compute initial expression from the ref weighted by cell-type counts from HoverNet segmentation
    This may differ by dataset; ideally refactor into a class. For now, implement a function for the HER2 dataset.
    :param count_dict:
    :param sincell_ref:
    :return:
    '''
    weight_dict = {}
    ref_celltype = ['neoplastic', 'inflammatory', 'connective','non-neoplastic epithelial']  
    ref_sum = sum([v for k, v in count_dict.items() if k in ref_celltype]) #record the total number of cells in the patch
    orig_exp = np.zeros((1,sincell_ref.shape[1])) #store the output result
    if ref_sum==0: #if no cells are detected on the patch
        for celltype in ref_celltype:
            orig_exp += np.array(sincell_ref[sincell_ref.index==celltype])*(1/len(ref_celltype)) #take the mean expression across each ref cell type
        return orig_exp
    else:
        for k, v in count_dict.items():
            if k in ref_celltype:
                weight_dict[k] = count_dict[k] / ref_sum #record each cell-type weight as count / total count
        for k,v in weight_dict.items():
            orig_exp += np.array(sincell_ref[sincell_ref.index==k])*weight_dict[k] #weighted sum of ref expressions to obtain the noise-exp
        return orig_exp

def gen_origExp_for_all_patch(path,ref_path,save_path,add_noise=False):
    id2tissue = {'0':'nolabe', #hover-net class labels
                 '1':'neoplastic',
                 '2':'inflammatory',
                 '3':'connective',
                 '4':'dead',
                 '5':'non-neoplastic epithelial'}
    ref = process_ref(ref_path)
    gene_num=len(ref.columns)
    for json_file in os.listdir(path): #process each patch in sequence
        spec_path = os.path.join(path,json_file)
        result = count_celltype(spec_path,id2tissue)
        ####You can add noise to the cell ratios to validate robustness
        if add_noise:
            noise_range = (-2, 2)  # Adjust the noise range; in this example it's [-2, 2]
            noisy_cell_types = {}
            for key, value in result.items():
                # randomly generate noise
                noise = random.randint(noise_range[0], noise_range[1])
                noisy_value = value + noise
                noisy_cell_types[key] = max(0, noisy_value)
            result=noisy_cell_types
        # #Prepare the single-cell ref file: select genes expressed in spatial transcriptomics; in the HER2+ dataset there are 785 genes
        orig_exp = create_orig_exp(result,ref)
        orig_exp = orig_exp.reshape((gene_num,))
        np.save(os.path.join(save_path,json_file.split('.')[0]+'.npy'),orig_exp)




if __name__ == '__main__':
    path = './04-patch-segResult/json/' #Hover-Net outputs path
    ref_path = './05-ref.csv' #ref path
    save_path = './06-noise_exp/' #save path
    gen_origExp_for_all_patch(path,ref_path,save_path)


