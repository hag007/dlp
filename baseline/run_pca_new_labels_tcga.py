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
from sklearn.model_selection import train_test_split

def knn(X_train,y_train, X_test, y_test):
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(X_train, y_train)
    score=neigh.score(X_test, y_test)
    print(score)

    return score

def main():
    print("start script...")
    dataset_names_tcga=constants.ALL_DATASET_NAMES
    dataset= datasets.Dataset(dataset_names=dataset_names_tcga, data_type=constants.DATA_TYPE)
    dataloader_ctor= datasets.DataLoader(dataset, 0.2, 0.2)
    trainloader = dataloader_ctor.train_loader()
    test_original_loader = dataloader_ctor.test_loader()

    dataset_names_new=constants.NEW_DATASETS_NAMES

    dataset= datasets.Dataset(dataset_names=dataset_names_new, data_type=constants.DATA_TYPE)
    dataloader_ctor= datasets.DataLoader(dataset, 0.0, 0.0)
    testloader = dataloader_ctor.train_loader()

    dataset_names=dataset_names_tcga+dataset_names_new



    datas_original=tensor([])
    labels_original=tensor([]).long()
    for batch_idx, (data, label) in enumerate(trainloader):
        datas_original=torch.cat((datas_original, data), 0)
        labels_original=torch.cat((labels_original, label), 0)

    n_tcga_unique_labels=len(dataset_names_tcga)

    datas_new=tensor([])
    labels_new=tensor([]).long()
    for batch_idx, (data, label) in enumerate(testloader):
        datas_new=torch.cat((datas_new, data), 0)
        labels_new=torch.cat((labels_new,label+n_tcga_unique_labels), 0)


    datas_original=datas_original.cpu().numpy()
    labels_original=labels_original.cpu().numpy()
    datas_new=datas_new.cpu().numpy()
    labels_new=labels_new.cpu().numpy()

    n_components=2
    print("start pca...")
    pca = PCA(n_components=2).fit(np.vstack((datas_original,datas_new)))
    X_original=pca.transform(datas_original)
    X_new=pca.transform(datas_new)
    print("start tsne...")
    X_train=X_original
    X_test= X_new
    y_train=labels_original
    y_test=labels_new

    fig = plt.figure(1, figsize=(20, 20))
    ax = fig.add_subplot(111)

    X=np.vstack([X_train,X_test])
    xs=X[:, 0]
    ys=X[:, 1]
    labels=np.hstack([labels_original,labels_new])
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


    plt.savefig(os.path.join(constants.OUTPUT_GLOBAL_DIR, "pca_tcga_new_labels.png"))
    plt.clf()
    plt.subplots(1,figsize=(10,10))
    with torch.no_grad():

        data_train=tensor([])
        labels_train=tensor([]).long()
        for batch_idx, (data, label) in enumerate(trainloader):
            data_train=torch.cat((data_train, data), 0)
            labels_train=torch.cat((labels_train, label), 0)

        data_test_original=tensor([])
        labels_test=tensor([]).long()
        for batch_idx, (data, label) in enumerate(test_original_loader):
            data_original_test=torch.cat((data_test_original, data), 0)
            labels_original_test=torch.cat((labels_test, label), 0)

        data_test=tensor([])
        labels_test=tensor([]).long()
        for batch_idx, (data, label) in enumerate(testloader):
            data_test=torch.cat((data_test, data), 0)
            labels_test=torch.cat((labels_test, label), 0)

        pca = pca.fit(data_train.cpu().numpy())
        n_labels = len(dataset_names)
        X_train=pca.transform(data_train.cpu().numpy())
        y_train=labels_train.cpu().numpy()
        X_original_test=pca.transform(data_original_test.cpu().numpy())
        y_original_test=labels_original_test.cpu().numpy()
        X_test=pca.transform(data_test.cpu().numpy())
        y_test=labels_test.cpu().numpy() + n_labels

        y_original=knn(X_train,y_train, X_original_test, y_original_test)

        ys=[]
        fractions=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
        for fraction in fractions:
            new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(X_test, y_test, test_size=1-fraction)
            ys.append(knn(np.vstack((X_train,new_X_train)),np.concatenate((y_train,new_y_train)), new_X_test, new_y_test))

        plt.plot(fractions, ys, label="knn score of new labels")
        plt.plot([0.01, 0.99], [y_original,y_original], label="knn score of original labels")
        plt.xlabel("test fraction")
        plt.ylabel("knn score")
        plt.legend()
        plt.savefig(os.path.join(constants.OUTPUT_GLOBAL_DIR, "tcga_new_label_plot.png"))

if __name__=='__main__':
    main()