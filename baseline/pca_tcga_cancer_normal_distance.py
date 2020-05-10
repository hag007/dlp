import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import  constants_tcga as constants
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datasets import datasets

import matplotlib.cm as cm
import matplotlib.colors as ml_colors
from matplotlib.lines import Line2D
import torch
from torch import tensor
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from scipy.spatial import distance_matrix
import seaborn as sns

def plot_median_diff(test_loader, dataset_names):
    zs=tensor([])
    labels=tensor([]).long()

    for batch_idx, (data, label) in enumerate(test_loader):
        data=data.cpu().numpy()
        pca = PCA(n_components=100).fit(data)
        zs=pca.transform(data)
        zs = TSNE(n_components=2, perplexity=30).fit_transform(zs)
    labels=label.cpu().numpy()

    zs[np.isneginf(zs)]=-1000000000
    zs[np.isposinf(zs)]=1000000000

    xs,ys=list(zip(*zs))
    label_unique = np.arange(len(dataset_names))

    cancer_dict_tumor={}
    cancer_dict_normal={}
    for a in label_unique:
        if "tumor" in dataset_names[a]:
            cancer_dict_tumor[dataset_names[a].split("_")[0]]=np.array([np.median([xs[i] for i, b in enumerate(labels) if a==b]), np.median([ys[i] for i, b in enumerate(labels) if a==b])])
        else:
            cancer_dict_normal[dataset_names[a].split("_")[0]]=np.array([np.median([xs[i] for i, b in enumerate(labels) if a==b]), np.median([ys[i] for i, b in enumerate(labels) if a==b])])

    df=pd.DataFrame()
    for i, k in enumerate(cancer_dict_tumor):
        try:
            diff=cancer_dict_tumor[k]-cancer_dict_normal[k]
        except:
            continue
        df[k]=diff


    df=df.T
    dm=distance_matrix(df, df)
    sns.heatmap(pd.DataFrame(data=dm, index=df.index))
    plt.savefig(os.path.join(constants.OUTPUT_GLOBAL_DIR, "hc11.png"))
    plt.clf()

    sns.clustermap(pd.DataFrame(data=dm, index=df.index, columns=df.index), method='single')
    plt.savefig(os.path.join(constants.OUTPUT_GLOBAL_DIR, "hc22.png"))
    plt.clf()


def main(model,use_z, fraction, epoch_checkpoint=300, suffix=""):

    n_latent_layer=2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset_names=constants.ALL_DATASET_NAMES
    dataset= datasets.Dataset(dataset_names, "tcga")
    dataloader_ctor= datasets.DataLoader(dataset, 0.0, 0.0)
    trainloader = dataloader_ctor.train_loader(batch_size=dataset.__len__())

    with torch.no_grad():
        patches_tcga=plot_median_diff(trainloader, dataset_names)
        plt.legend(handles=patches_tcga)


if __name__=="__main__":


    fractions=[1.0] # , 0.33, 0.67, 1.0]
    use_zs=[True] # , False]
    models=[constants.MODEL_VAE]

    epoch_checkpoints=[150] # [50, 100, 150,200,250,300]
    suffix="_min"
    for cur_use_z in use_zs:
        for cur_fraction in fractions:
            for model in models:
                for cur_epoch_checkpoint in epoch_checkpoints:
                    print("start {} {} use_z={}".format(cur_fraction, model, cur_use_z))
                    # params.append([main, [model, cur_use_z, cur_fraction, epoch_checkpoint, "_min"]])
                    main(model=model, use_z=cur_use_z, fraction=cur_fraction, epoch_checkpoint=cur_epoch_checkpoint, suffix=suffix)
