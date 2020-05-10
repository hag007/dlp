import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch.nn import functional as F
from torch import tensor

from nn.models import Encoder, Decoder, Classifier
import constants_tcga as constants
from datasets import datasets
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.cm as cm
import matplotlib.colors as ml_colors
from matplotlib.lines import Line2D
from sklearn.neighbors import KNeighborsClassifier

limit=3

def knn(X_train,y_train, X_test, y_test):
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(X_train, y_train)
    score=neigh.score(X_test, y_test)
    print(score)

    return score

def plot(model, test_loader, device, dataset_names, colormap, bg_color):
    zs=tensor([])
    mus=tensor([])
    logvars=tensor([])
    labels=tensor([]).long()

    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(device)
        z, mu, logvar, _ = model(data)
        zs=torch.cat((zs, z), 0)
        mus=torch.cat((mus, mu), 0)
        logvars=torch.cat((logvars, logvar), 0)
        labels=torch.cat((labels, label), 0)

    zs=zs.cpu().numpy()
    labels=labels.cpu().numpy()

    X_pca=zs
    labels = [labels[i] for i, a in enumerate(X_pca) if np.abs(a[0]) <limit and np.abs(a[1]) <limit]
    X_pca=[a for a in X_pca if np.abs(a[0]) <limit and np.abs(a[1]) <limit]
    # X_pca = PCA(n_components=2).fit_transform(zs)

    xs,ys=list(zip(*X_pca))

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
                    bbox=dict(boxstyle='round,pad=0.5', fc=bg_color, alpha=0.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=3, headlength=2))


    return X_pca, labels, patches


def main(model,use_z, fraction, epoch_checkpoint=300, suffix=""):

    n_latent_layer=2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset_names_icgc=constants.ICGC_ALL_DATASET_NAMES
    dataset= datasets.Dataset(dataset_names_icgc, "icgc")
    dataloader_ctor= datasets.DataLoader(dataset, 0.0, 0.0)
    testloader = dataloader_ctor.train_loader()

    dataset_names_tcga=constants.ALL_DATASET_NAMES
    dataset= datasets.Dataset(dataset_names_tcga, "tcga")
    dataloader_ctor= datasets.DataLoader(dataset, 0.2, 0.2)
    trainloader = dataloader_ctor.train_loader()

    encoder=Encoder(n_latent_layer=n_latent_layer)
    decoder=Decoder(n_latent_layer=n_latent_layer)

    path_format_to_save=os.path.join(constants.CACHE_GLOBAL_DIR, constants.DATA_TYPE, "model_{}_{}_{}_{{}}".format(fraction,model,"z" if use_z else "mu"))
    PATH_ENCODER= os.path.join(path_format_to_save,"ENC_mdl")
    PATH_DECODER= os.path.join(path_format_to_save,"DEC_mdl")

    load_model=True
    if load_model and os.path.exists(PATH_ENCODER.format(epoch_checkpoint)+suffix):
        encoder.load_state_dict(torch.load(PATH_ENCODER.format(epoch_checkpoint)+suffix))
        encoder.eval()
        decoder.load_state_dict(torch.load(PATH_DECODER.format(epoch_checkpoint)+suffix))
        decoder.eval()

    with torch.no_grad():
        path_to_save=path_format_to_save.format(epoch_checkpoint)
        plt.subplots(figsize=(20,20))
        colormap = cm.jet
        zs_train, labels_train, patches_tcga=plot(encoder, trainloader, device, constants.ALL_DATASET_NAMES, colormap, 'yellow')
        plt.legend(handles=patches_tcga)
        plt.savefig(os.path.join(path_to_save, "zs_scatter{}.png".format(suffix + "_tcga")))
        n_tcga_unique_labels=len(dataset_names_tcga)

        colormap = cm.terrain
        zs_test, labels_test, patches_icgc =plot(encoder, testloader, device, constants.ICGC_ALL_DATASET_NAMES, colormap, 'blue')
        plt.legend(handles=patches_tcga+patches_icgc)
        plt.savefig(os.path.join(path_to_save, "zs_scatter{}.png".format(suffix + "_icgc")))

        X_train=zs_train
        X_test= zs_test
        y_train=labels_train
        y_test=[constants.ICGC_PSEUDO_LABELS[constants.ICGC_DATASETS_NAMES[a]] for a in labels_test]
        knn(X_train,y_train, X_test, y_test)


if __name__=="__main__":
    filter_func_dict={
        0.01:lambda a: a % 100 == 0,
        0.05:lambda a: a % 20 == 0,
        0.1:lambda a: a % 10 == 0,
        0.33:lambda a: a % 3 == 0,
        0.67:lambda a: a % 3 > 0,
        1.0:lambda a: True
    }

    fractions=[1.0] # , 0.33, 0.67, 1.0]
    use_zs=[True] # , False]
    models=[constants.MODEL_VAE]

    epoch_checkpoints=[2000] # [50, 100, 150,200,250,300]
    suffix="_min"
    for cur_use_z in use_zs:
        for cur_fraction in fractions:
            for model in models:
                for cur_epoch_checkpoint in epoch_checkpoints:
                    print("start {} {} use_z={}".format(cur_fraction, model, cur_use_z))
                    # params.append([main, [model, cur_use_z, cur_fraction, epoch_checkpoint, "_min"]])
                    main(model=model, use_z=cur_use_z, fraction=cur_fraction, epoch_checkpoint=cur_epoch_checkpoint, suffix=suffix)

    # p.map(func_star, params)
