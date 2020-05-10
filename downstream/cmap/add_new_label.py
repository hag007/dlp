import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch.nn import functional as F
from torch import tensor

from nn.models import Encoder, Decoder, Classifier
from plots.scatter_plot_test import plot
import constants_cmap
from datasets import datasets
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.cm as cm
import matplotlib.colors as ml_colors
from matplotlib.lines import Line2D
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

def knn(X_train,y_train, X_test, y_test):
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(X_train, y_train)
    score=neigh.score(X_test, y_test)
    print(score)

    return score


def plot_bu(model, test_loader, device, suffix, path_to_save, dataset_names, colormap, bg_color):
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

    # xs,ys=list(zip(*zs.cpu().numpy()))
    zs=zs.cpu().numpy()
    labels=labels.cpu().numpy()

    # np.save(os.path.join(path_to_save, "latent_features{}.npy".format(suffix)), np.hstack([zs, labels.reshape(-1,1), [[dataset_names[a]] for a in labels]]))
    X_pca = zs # PCA(n_components=2).fit_transform(zs)

    limit=10
    labels=np.array([labels[i] for i, a in enumerate(zs) if np.abs(a[0])<limit and np.abs(a[1])<limit])
    zs=[a for a in zs if np.abs(a[0])<limit and np.abs(a[1])<limit]

    xs,ys=list(zip(*zs))

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
    return patches

def main(model,use_z, fraction, epoch_checkpoint=300, suffix=""):

    n_latent_layer=2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    genes=None
    genes_name=None
    # genes=np.load("/media/hag007/Data/dlproj/cache_global/datasets/vemurafenib_resveratrol_olaparib/genes.npy", allow_pickle=True)
    # genes_name="vemurafenib_resveratrol_olaparib"

    dataset_names_new=constants_cmap.NEW_DATASETS_NAMES
    dataset= datasets.Dataset(dataset_names_new, constants_cmap.DATA_TYPE)
    dataloader_ctor= datasets.DataLoader(dataset, 0.0, 0.0)
    testloader = dataloader_ctor.train_loader()

    dataset_names=constants_cmap.DATASETS_NAMES
    dataset= datasets.Dataset(dataset_names, constants_cmap.DATA_TYPE)
    dataloader_ctor= datasets.DataLoader(dataset, 0.2, 0.2)
    trainloader = dataloader_ctor.train_loader()
    test_original_loader = dataloader_ctor.test_loader()

    encoder=Encoder(n_latent_layer=n_latent_layer)
    decoder=Decoder(n_latent_layer=n_latent_layer)
    classifier=Classifier(n_input_layer=n_latent_layer, n_classes=(len(constants_cmap.DATASETS_FILES) if genes_name is None else genes_name.count("_") + 1))

    path_format_to_save=os.path.join(constants_cmap.CACHE_GLOBAL_DIR, constants_cmap.DATA_TYPE, "model_{}_{}_{}_{{}}".format(fraction, model, "z" if use_z else "mu"))
    PATH_ENCODER= os.path.join(path_format_to_save,"ENC_mdl")
    PATH_DECODER= os.path.join(path_format_to_save,"DEC_mdl")
    PATH_CLASSIFIER= os.path.join(path_format_to_save,"CLS_mdl")

    load_model=True
    if load_model and os.path.exists(PATH_ENCODER.format(epoch_checkpoint)+suffix):
        encoder.load_state_dict(torch.load(PATH_ENCODER.format(epoch_checkpoint)+suffix))
        encoder.eval()
        decoder.load_state_dict(torch.load(PATH_DECODER.format(epoch_checkpoint)+suffix))
        decoder.eval()
        if model != constants_cmap.MODEL_VAE:
            classifier.load_state_dict(torch.load(PATH_CLASSIFIER.format(epoch_checkpoint)+suffix))
            classifier.eval()

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

        n_labels = len(dataset_names)
        X_train=encoder(data_train)[0].cpu().numpy()
        y_train=labels_train.cpu().numpy()
        X_original_test=encoder(data_original_test)[0].cpu().numpy()
        y_original_test=labels_original_test.cpu().numpy()
        X_test=encoder(data_test)[0].cpu().numpy()
        y_test=labels_test.cpu().numpy() + n_labels

        y_original=knn(X_train,y_train, X_original_test, y_original_test)

        ys=[]
        fractions= [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
        new_X_test=X_test
        new_y_test=y_test
        # knn(X_train,y_train, new_X_test, new_y_test)
        for fraction in fractions:
            new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(X_test, y_test, test_size=1.0-fraction)
            ys.append(knn(np.vstack((X_train,new_X_train)),np.concatenate((y_train,new_y_train)), new_X_test, new_y_test))
        plt.plot(fractions, ys, label="knn score of new labels")
        plt.plot([0.01, 0.99], [y_original,y_original], label="knn score of original labels")
        plt.xlabel("test fraction")
        plt.ylabel("knn score")
        plt.legend()
        plt.savefig(os.path.join(constants_cmap.OUTPUT_GLOBAL_DIR, "new_label_plot.png"))

        path_to_save=path_format_to_save.format(epoch_checkpoint)
        plt.subplots(figsize=(20,20))
        colormap = cm.jet
        patches_tcga=plot_bu(encoder, trainloader, device, suffix + "_tcga", path_to_save, constants_cmap.DATASETS_NAMES, colormap, 'yellow')
        plt.legend(handles=patches_tcga)
        plt.savefig(os.path.join(path_to_save, "zs_scatter{}.png".format(suffix + "_tcga")))

        colormap = cm.terrain
        patches_new=plot_bu(encoder, testloader, device, suffix + "_new", path_format_to_save.format(epoch_checkpoint), constants_cmap.NEW_DATASETS_NAMES, colormap, 'blue')
        plt.legend(handles=patches_tcga+patches_new)
        plt.savefig(os.path.join(path_to_save, "zs_scatter{}.png".format(suffix + "_new")))


if __name__=="__main__":

    fractions=[0.0] # , 0.33, 0.67, 1.0]
    use_zs=[True] # , False]
    models=[constants_cmap.MODEL_VAE]

    epoch_checkpoints=[2000] # [50, 100, 150,200,250,300]
    suffix=""
    for cur_use_z in use_zs:
        for cur_fraction in fractions:
            for model in models:
                for cur_epoch_checkpoint in epoch_checkpoints:
                    print("start {} {} use_z={}".format(cur_fraction, model, cur_use_z))
                    # params.append([main, [model, cur_use_z, cur_fraction, epoch_checkpoint, "_min"]])
                    main(model=model, use_z=cur_use_z, fraction=cur_fraction, epoch_checkpoint=cur_epoch_checkpoint, suffix=suffix)

    # p.map(func_star, params)
