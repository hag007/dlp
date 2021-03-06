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

def knn(X_train,y_train, X_test, y_test):
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(X_train, y_train)
    score=neigh.score(X_test, y_test)
    print(score)
    print(neigh.score(X_train, y_train))
    return score

def main():
    print("start script...")
    dataset_names_tcga=constants.ALL_DATASET_NAMES
    dataset= datasets.Dataset(dataset_names=dataset_names_tcga, data_type="tcga")
    dataloader_ctor= datasets.DataLoader(dataset, 0.2, 0.2)
    trainloader = dataloader_ctor.train_loader()
    testloader = dataloader_ctor.test_loader()
    dataset_names=dataset_names_tcga
    datas=tensor([])
    labels=tensor([]).long()

    for batch_idx, (data, label) in enumerate(trainloader):
        datas=torch.cat((datas, data), 0)
        labels=torch.cat((labels, label), 0)

    n_samples_tcga=datas.shape[0]
    n_tcga_unique_labels=len(dataset_names_tcga)

    for batch_idx, (data, label) in enumerate(testloader):
        datas=torch.cat((datas, data), 0)
        labels=torch.cat((labels, label), 0)

    datas=datas.cpu().numpy()
    labels=labels.cpu().numpy()

    n_components=2000
    print("start pca...")
    X_pca = datas # PCA(n_components=n_components).fit_transform(datas)
    print("start tsne...")
    X = TSNE(n_components=2, metric="euclidean", perplexity=30.0).fit_transform(X_pca)
    X_train=X_pca[:n_samples_tcga]
    X_test= X_pca[n_samples_tcga:]
    y_train=labels[:n_samples_tcga]
    y_test=labels[n_samples_tcga:] # [constants.ICGC_PSEUDO_LABELS[a-n_tcga_unique_labels] for a in labels[n_samples_tcga:]]
    knn(X_train,y_train, X_test, y_test)
    fig = plt.figure(1, figsize=(20, 20))
    ax = fig.add_subplot(111)
    xs=X[:, 0]
    ys=X[:, 1]
    ax.scatter(xs, ys, c=labels)
    colormap = cm.jet
    plt.scatter(xs,ys, c=[a for a in labels], cmap=colormap) # sns.color_palette("Paired", n_colors=len(constants.DATASETS_INCLUDED))[a]


    label_unique = np.arange(len(np.unique(labels)))
    colorlist_unique = [ml_colors.rgb2hex(colormap(a)) for a in
                    label_unique / float(max(labels))]
    patches = [Line2D([0], [0], marker='o', color='gray', label=dataset_names[a],
                  markerfacecolor=c) for a, c in zip(label_unique, colorlist_unique)]

    for a in label_unique:
        plt.scatter([np.median([xs[i] for i, b in enumerate(labels) if a==b])],[np.median([ys[i] for i, b in enumerate(labels) if a==b])], s=2000, c=colorlist_unique[a], cmap=colormap, alpha=0.5)
        plt.annotate(dataset_names[a],
                    xy=(np.median([xs[i] for i, b in enumerate(labels) if a==b]), np.median([ys[i] for i, b in enumerate(labels) if a==b])), xytext=(-20, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc=('yellow' if a<n_tcga_unique_labels else 'blue') , alpha=0.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=3, headlength=2))

    plt.legend(handles=patches)


    plt.savefig(os.path.join(constants.OUTPUT_GLOBAL_DIR, "clustering_tnse_tcga_knn.png"))

if __name__=='__main__':
    main()