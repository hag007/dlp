import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch.nn import functional as F
from torch import tensor
import time
from nn.models import Encoder, Decoder, Classifier
import constants_tcga as constants
from datasets import datasets
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as ml_colors
from matplotlib.lines import Line2D

filter_func_dict={
        0.01:lambda a: a % 100 == 0,
        0.05:lambda a: a % 20 == 0,
        0.1:lambda a: a % 10 == 0,
        0.33:lambda a: a % 3 == 0,
        0.67:lambda a: a % 3 > 0,
        1.0:lambda a: True
    }

def plot_median_diff(model, test_loader, device, suffix, path_to_save, dataset_names, epoch_checkpoint, colormap, bg_color):
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

    # np.save(os.path.join(path_to_save, "latent_features{}.npy".format(suffix)), np.hstack([zs, labels.reshape(-1,1), [[dataset_names[a]] for a in labels]]))

    xs,ys=list(zip(*zs))
    label_unique = np.arange(len(dataset_names))

    cancer_dict_tumor={}
    cancer_dict_normal={}
    for a in label_unique:
        # plt.scatter([np.median([xs[i] for i, b in enumerate(labels) if a==b])],[np.median([ys[i] for i, b in enumerate(labels) if a==b])], s=2000, c=colorlist_unique[a], cmap=colormap, alpha=0.5)
        # plt.annotate(dataset_names[a],
        #             xy=(np.median([xs[i] for i, b in enumerate(labels) if a==b]), np.median([ys[i] for i, b in enumerate(labels) if a==b])), xytext=(-20, 20), textcoords='offset points',
        #             bbox=dict(boxstyle='round,pad=0.5', fc=bg_color, alpha=0.5),
        #             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=3, headlength=2))
        if "tumor" in dataset_names[a]:
            cancer_dict_tumor[dataset_names[a].split("_")[0]]=np.array([np.median([xs[i] for i, b in enumerate(labels) if a==b]), np.median([ys[i] for i, b in enumerate(labels) if a==b])])
        else:
            cancer_dict_normal[dataset_names[a].split("_")[0]]=np.array([np.median([xs[i] for i, b in enumerate(labels) if a==b]), np.median([ys[i] for i, b in enumerate(labels) if a==b])])


    labels_unique_half=np.arange(int(np.ceil(len(dataset_names)/2.0)))
    colorlist_unique = [ml_colors.rgb2hex(colormap(a)) for a in
                    labels_unique_half / float(max(labels)/2)]
    patches = [Line2D([0], [0], marker='o', color='gray', label=dataset_names[a*2].split("_")[0],
                  markerfacecolor=c) for a, c in zip(labels_unique_half, colorlist_unique)]

    df=pd.DataFrame()
    for i, k in enumerate(cancer_dict_tumor):
        try:
            diff=cancer_dict_tumor[k]-cancer_dict_normal[k]
        except:
            continue
        df[k]=diff
        plt.scatter([diff[0]],[diff[1]], s=2000, c=colorlist_unique[i], cmap=colormap, alpha=0.5)
        plt.annotate(k+"_diff",
                    xy=(diff[0], diff[1]), xytext=(-20, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc=bg_color, alpha=0.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=3, headlength=2))


    df.to_csv(os.path.join(constants.OUTPUT_GLOBAL_DIR, "diff_{}{}_{}.tsv".format(epoch_checkpoint,suffix, time.time())), sep='\t')



    return patches

def main(model,use_z, fraction, epoch_checkpoint=300, suffix=""):

    filter_func=filter_func_dict[fraction]

    n_latent_layer=2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    genes=None
    genes_name=None
    # genes=np.load("/media/hag007/Data/dlproj/cache_global/datasets/vemurafenib_resveratrol_olaparib/genes.npy", allow_pickle=True)
    # genes_name="vemurafenib_resveratrol_olaparib"

    dataset_names=constants.ALL_DATASET_NAMES
    dataset= datasets.Dataset(dataset_names, "tcga")
    dataloader_ctor= datasets.DataLoader(dataset, 0.0, 0.0)
    trainloader = dataloader_ctor.train_loader()

    encoder=Encoder(n_latent_layer=n_latent_layer)
    decoder=Decoder(n_latent_layer=n_latent_layer)
    classifier=Classifier(n_input_layer=n_latent_layer, n_classes=len(dataset_names))

    path_format_to_save=os.path.join(constants.CACHE_GLOBAL_DIR, constants.DATA_TYPE, "model_{}_{}_{}_{{}}".format(fraction,model,"z" if use_z else "mu"))
    PATH_ENCODER= os.path.join(path_format_to_save,"ENC_mdl")
    PATH_DECODER= os.path.join(path_format_to_save,"DEC_mdl")
    PATH_CLASSIFIER= os.path.join(path_format_to_save,"CLS_mdl")

    load_model=True
    if load_model and os.path.exists(PATH_ENCODER.format(epoch_checkpoint)+suffix):
        encoder.load_state_dict(torch.load(PATH_ENCODER.format(epoch_checkpoint)+suffix))
        encoder.eval()
        decoder.load_state_dict(torch.load(PATH_DECODER.format(epoch_checkpoint)+suffix))
        decoder.eval()
        if model != constants.MODEL_VAE:
            classifier.load_state_dict(torch.load(PATH_CLASSIFIER.format(epoch_checkpoint)+suffix))
            classifier.eval()

    with torch.no_grad():
        path_to_save=path_format_to_save.format(epoch_checkpoint)
        # plt.subplots(figsize=(20,20))
        # colormap = cm.jet
        # patches_tcga=plot(encoder, trainloader, device, suffix + "_tcga", path_to_save, constants.DATASETS_NAMES, colormap, 'yellow')
        # # plt.legend(handles=patches_tcga)
        # # plt.savefig(os.path.join(path_to_save, "zs_scatter{}.png".format(suffix + "_tcga_diff")))
        # plt.clf()

        plt.subplots(figsize=(20,20))
        colormap = cm.jet
        patches_tcga=plot_median_diff(encoder, trainloader, device, suffix + "_tcga", path_to_save, dataset_names, epoch_checkpoint, colormap, 'yellow')
        plt.legend(handles=patches_tcga)
        plt.savefig(os.path.join(path_to_save, "zs_scatter{}_{}.png".format(suffix + "_tcga_diff", time.time())))


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

    # p.map(func_star, params)
