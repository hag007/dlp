import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import  constants_tcga as constants
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datasets import cmap_datasets

import matplotlib.cm as cm
import matplotlib.colors as ml_colors
from matplotlib.lines import Line2D
import torch
from torch import tensor
import numpy as np

def main():
    print("start tsne...")
    dataset_names=[a for i, a in enumerate(constants.DATASETS_NAMES) if constants.DATASETS_INCLUDED[i]]
    dataset= cmap_datasets.CMAPDataset(dataset_names=dataset_names, data_type="tcga")
    dataloader_ctor= cmap_datasets.CMAPDataLoader(dataset, 0.2, 0.2)
    trainloader = dataloader_ctor.train_loader()
    datas=tensor([])
    labels=tensor([]).long()


    for batch_idx, (data, label) in enumerate(trainloader):
        datas=torch.cat((datas, data), 0)
        labels=torch.cat((labels, label), 0)


    datas=datas.cpu().numpy()
    labels=labels.cpu().numpy()

    n_components=2
    X_pca = PCA(n_components=10).fit_transform(datas)
    X = TSNE(n_components=n_components, metric="euclidean", perplexity=15.0).fit_transform(X_pca)
    fig = plt.figure(1, figsize=(20, 20))
    ax = fig.add_subplot(111)
    xs=X[:, 0]
    ys=X[:, 1]
    ax.scatter(xs, ys, c=labels)
    colormap = cm.jet
    plt.scatter(xs,ys, c=[a for a in labels], cmap=colormap) # sns.color_palette("Paired", n_colors=len(constants.DATASETS_INCLUDED))[a]


    label_unique = np.arange(len(dataset_names))
    colorlist_unique = [ml_colors.rgb2hex(colormap(a)) for a in
                    label_unique / float(max(labels))]
    patches = [Line2D([0], [0], marker='o', color='gray', label=dataset_names[a],
                  markerfacecolor=c) for a, c in zip(label_unique, colorlist_unique)]

    for a in label_unique:
        plt.scatter([np.median([xs[i] for i, b in enumerate(labels) if a==b])],[np.median([ys[i] for i, b in enumerate(labels) if a==b])], s=2000, c=colorlist_unique[a], cmap=colormap, alpha=0.5)
        plt.annotate(dataset_names[a],
                    xy=(np.median([xs[i] for i, b in enumerate(labels) if a==b]), np.median([ys[i] for i, b in enumerate(labels) if a==b])), xytext=(-20, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=3, headlength=2))

    plt.legend(handles=patches)


    plt.savefig(os.path.join(constants.OUTPUT_GLOBAL_DIR, "tnse.png"))

if __name__=='__main__':
    main()