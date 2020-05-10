import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import tensor
import pandas as pd
import os
from torch.utils.data.sampler import SubsetRandomSampler
import constants_cmap
from scipy.stats import zscore, rankdata
import shutil


constants_cmap.DATASETS_DIR= os.path.join(constants_cmap.BASE_PROFILE, "tcga_datasets")

CANCER_TYPES= ["LUSC", "KIRC" , "KIRP", "LUSC", "LUAD", "COAD", "BRCA", "STAD", "LIHC", "READ", "PRAD", "BLCA", "HNSC", "THCA", "UCEC", "OV", "PAAD"]
constants_cmap.DATASETS_FILES=[
    "BLCA_normal_19.tsv",
    "BLCA_tumor_411.tsv",
    "BRCA_normal_113.tsv",
    "BRCA_tumor_1097.tsv",
    "COAD_normal_41.tsv",
    "COAD_tumor_469.tsv",
    "BLCA_normal_19.tsv",
    "BLCA_tumor_411.tsv",
    "COAD_tumor_469.tsv",
    "HNSC_normal_44.tsv",
    "HNSC_tumor_500.tsv",
    "KIRC_normal_72.tsv",
    "KIRC_tumor_534.tsv",
    "KIRP_normal_32.tsv",
    "KIRP_tumor_288.tsv",
    "LIHC_normal_50.tsv",
    "LIHC_tumor_371.tsv",
    "LUAD_normal_59.tsv",
    "LUAD_tumor_524.tsv",
    "LUSC_normal_49.tsv",
    "LUSC_tumor_501.tsv",
    "OV_normal_0.tsv",
    "OV_tumor_374.tsv",
    "PAAD_normal_4.tsv",
    "PAAD_tumor_177.tsv",
    "PRAD_normal_52.tsv",
    "PRAD_tumor_498.tsv",
    "READ_normal_10.tsv",
    "READ_tumor_166.tsv",
    "STAD_normal_32.tsv",
    "STAD_tumor_375.tsv",
    "THCA_normal_58.tsv",
    "THCA_tumor_502.tsv",
    "UCEC_normal_35.tsv",
    "UCEC_tumor_547.tsv"
]


constants_cmap.DATASETS_NAMES=["_".join(a.split("_")[0:2]) for a in constants_cmap.DATASETS_FILES]

constants_cmap.DATASETS_F=[0 for a in constants_cmap.DATASETS_FILES]
constants_cmap.DATASETS_F_NAMES=["nib" for a in constants_cmap.DATASETS_FILES]


dataset_files=[a for i, a in enumerate(constants_cmap.DATASETS_FILES)]



dataset_names=[a for i, a in enumerate(constants_cmap.DATASETS_NAMES)]
dataset_files=[a for i, a in enumerate(constants_cmap.DATASETS_FILES)]
dataset_f=[a for i, a in enumerate(constants_cmap.DATASETS_F)]
dataset_f_names=[a for i, a in enumerate(constants_cmap.DATASETS_F_NAMES)]

path_to_cache=os.path.join(constants_cmap.CACHE_GLOBAL_DIR, "datasets", "tcga")
n_input_layer=2000
all_datasets=[pd.read_csv(os.path.join(constants_cmap.DATASETS_DIR, ds), sep='\t', index_col=0) for ds in dataset_files]

i=0
all_datasets_normalized=[]
for df, name in zip(all_datasets, dataset_names):
    indices= df.index
    # df=pd.DataFrame(data=zscore(cur_ds), index=indices, columns=genes)
    # df=df.apply(lambda a: (a-min(a))/(max(a)-min(a)))
    # df = df.divide(df.max(axis=1), axis=0)

    all_datasets_normalized.append(df)

all_datasets=all_datasets_normalized

df_X_all = pd.concat(all_datasets, axis=0).astype(np.float)
df_X_all.dropna(axis=1)

genes= df_X_all.columns
indices= df_X_all.index
df_X_all = np.apply_along_axis(lambda a: (a-min(a))/(max(max(a)-min(a), 0.0001)), 0, zscore(df_X_all))
df_X_all=pd.DataFrame(df_X_all, columns=genes)
df_X_all=df_X_all.dropna(axis=1)
genes= df_X_all.columns
df_X_all=df_X_all.values
genes_i=rankdata(df_X_all.var(axis=0))>df_X_all.shape[1]-n_input_layer
genes=genes[genes_i]
df_X_all =df_X_all[:,genes_i]
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




