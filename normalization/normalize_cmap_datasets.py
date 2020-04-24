import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import tensor
import pandas as pd
import os
from torch.utils.data.sampler import SubsetRandomSampler
import constants
from scipy.stats import zscore, rankdata
import shutil


dataset_names=[a for i, a in enumerate(constants.DATASETS_NAMES)]
dataset_files=[a for i, a in enumerate(constants.DATASETS_FILES)]
dataset_f=[a for i, a in enumerate(constants.DATASETS_F)]
dataset_f_names=[a for i, a in enumerate(constants.DATASETS_F_NAMES)]

path_to_cache=os.path.join(constants.CACHE_GLOBAL_DIR, "datasets", "cmap")

all_datasets=[pd.read_csv(os.path.join(constants.DATASETS_DIR, ds), sep='\t', index_col=0).T.drop(
        ["pr_gene_symbol", "pr_gene_symbol.1"]) for ds in dataset_files]

df_X_all = pd.concat(all_datasets, axis=0).astype(np.float)

genes= df_X_all.columns
df_X_all = np.apply_along_axis(lambda a: (a-min(a))/(max(a)-min(a)), 0, zscore(df_X_all))
genes=genes[rankdata(df_X_all.var(axis=0))>df_X_all.shape[1]-2000]
df_X_all =df_X_all[:,rankdata(df_X_all.var(axis=0))>df_X_all.shape[1]-2000]
df_y_all = [i for i, a in enumerate(all_datasets) for b in np.arange(a.shape[0])]
df_y_all_names = [dataset_names[i] for i, a in enumerate(all_datasets) for b in np.arange(a.shape[0])]
df_y_f = [dataset_f[i] for i, a in enumerate(all_datasets) for b in np.arange(a.shape[0])]
df_y_f_names = [dataset_f_names[i] for i, a in enumerate(all_datasets) for b in a]


start_i=0
for cur_ds, name in zip(all_datasets,dataset_names):

    try:
        os.makedirs(os.path.join(path_to_cache, name))
    except Exception:
        pass

    X=df_X_all[start_i:start_i+cur_ds.shape[0]]
    y=df_y_all[start_i:start_i+cur_ds.shape[0]]
    y_names=df_y_all_names[start_i:start_i+cur_ds.shape[0]]
    y_f=df_y_f[start_i:start_i+cur_ds.shape[0]]
    y_f_names=df_y_f_names[start_i:start_i+cur_ds.shape[0]]

    np.save(os.path.join(path_to_cache, name, "X.npy"), X)
    np.save(os.path.join(path_to_cache, name, "y.npy"), y)
    np.save(os.path.join(path_to_cache, name, "y_names.npy"), y_names)
    np.save(os.path.join(path_to_cache, name, "y_f.npy"), y_f)
    np.save(os.path.join(path_to_cache, name, "y_f_names.npy"), y_f_names)

    start_i+=cur_ds.shape[0]

np.save(os.path.join(path_to_cache, "genes.npy"), genes)




