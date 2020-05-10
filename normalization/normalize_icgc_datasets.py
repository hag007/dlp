import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import tensor
import pandas as pd
import os
from torch.utils.data.sampler import SubsetRandomSampler
import constants_tcga as constants
from scipy.stats import zscore, rankdata
import shutil


constants.DATASETS_DIR= os.path.join(constants.BASE_PROFILE, "icgc_datasets")

constants.DATASETS_FILES=constants.ICGC_DATASETS_FILES
constants.DATASETS_NAMES=constants.ICGC_DATASETS_NAMES
constants.DATASETS_F=constants.ICGC_DATASETS_F
constants.DATASETS_F_NAMES=constants.ICGC_DATASETS_F


dataset_files=[a for i, a in enumerate(constants.DATASETS_FILES)]
dataset_names=[a for i, a in enumerate(constants.DATASETS_NAMES)]
dataset_files=[a for i, a in enumerate(constants.DATASETS_FILES)]
dataset_f=[a for i, a in enumerate(constants.DATASETS_F)]
dataset_f_names=[a for i, a in enumerate(constants.DATASETS_F_NAMES)]

path_to_cache=os.path.join(constants.CACHE_GLOBAL_DIR, "datasets", "icgc")

all_datasets=[pd.read_csv(os.path.join(constants.DATASETS_DIR, ds), sep='\t', index_col=0) for ds in dataset_files]
all_datasets_normalized=[]

i=0
for cur_ds, name in zip(all_datasets, dataset_names):
    genes=np.load("/media/hag007/Data/dlproj/cache_global/datasets/tcga/genes.npy", allow_pickle=True)
    cur_ds=cur_ds.dropna(axis=1).reindex(genes, axis=1)
    indices= cur_ds.index
    df=pd.DataFrame(data=zscore(cur_ds), index=indices, columns=genes)
    df=df.apply(lambda a: (a-min(a))/(max(a)-min(a)))
    # df=cur_ds
    # df = df.divide(df.max(axis=1), axis=0)
    for idx in np.arange(df.shape[0]):
        median=np.nanmedian(df.iloc[idx,:])
        df.iloc[idx,:][np.isnan(df.iloc[idx,:])]=median
    all_datasets_normalized.append(df)

    try:
        os.makedirs(os.path.join(path_to_cache, name))
    except Exception:
        pass
    X=df.values
    y=[i for b in np.arange(df.shape[0])]
    y_names=[name for b in np.arange(df.shape[0])]
    y_f=[dataset_f[i]  for b in np.arange(df.shape[0])]
    y_f_names=[dataset_f_names[i]  for b in np.arange(df.shape[0])]

    np.save(os.path.join(path_to_cache, name, "X.npy"), X)
    np.save(os.path.join(path_to_cache, name, "y.npy"), y)
    np.save(os.path.join(path_to_cache, name, "y_names.npy"), y_names)
    np.save(os.path.join(path_to_cache, name, "y_f.npy"), y_f)
    np.save(os.path.join(path_to_cache, name, "y_f_names.npy"), y_f_names)
    i+=1

np.save(os.path.join(path_to_cache, "genes.npy"), genes)





