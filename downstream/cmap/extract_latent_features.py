import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch import tensor

from nn.models import Encoder, Decoder, Classifier
import constants_cmap
from datasets import datasets
import numpy as np

import matplotlib.cm as cm
import matplotlib.colors as ml_colors
from matplotlib.lines import Line2D

filter_func_dict={
        0.0:lambda a: False,
        0.005:lambda a: a % 200 == 0,
        0.01:lambda a: a % 100 == 0,
        0.02:lambda a: a % 50 == 0,
        0.03:lambda a: a % 33 ==0,
        0.04:lambda a: a % 25 == 0,
        0.05:lambda a: a % 20 == 0,
        0.1:lambda a: a % 10 == 0,
        0.2:lambda a: a % 5 == 0,
        0.33:lambda a: a % 3 == 0,
        0.67:lambda a: a % 3 > 0,
        0.5:lambda a: a % 2 == 0,
        1.0:lambda a: True
    }

def extract_latent_dimension(model, test_loader, device, suffix, path_to_save):

    dataset_names= constants_cmap.ALL_DATASET_NAMES # [a for i, a in enumerate(constants.DATASETS_NAMES) if constants.DATASETS_DICT[a]]

    zs=tensor([])
    mus=tensor([])
    logvars=tensor([])
    labels=tensor([]).long()
    reds=tensor([])

    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(device)
        z, mu, logvar, red = model(data)
        zs=torch.cat((zs, z), 0)
        mus=torch.cat((mus, mu), 0)
        logvars=torch.cat((logvars, logvar), 0)
        labels=torch.cat((labels, label), 0)
        reds=torch.cat((reds, red), 0)

    limit=100
    reds=reds.cpu().numpy()
    labels=labels.cpu().numpy()
    reds[np.isneginf(reds)]=-1000000000
    reds[np.isposinf(reds)]=1000000000

    labels=np.array([labels[i] for i, a in enumerate(reds) if np.abs(a[0])<limit and np.abs(a[1])<limit])
    reds=[a for a in reds if np.abs(a[0])<limit and np.abs(a[1])<limit]

    np.save(os.path.join(path_to_save, "latent_features{}.npy".format(suffix)), np.hstack([reds, labels.reshape(-1,1), [[dataset_names[a]] for a in labels]]))

    xs,ys=list(zip(*reds))
    plt.subplots(figsize=(20,20))
    plt.scatter(xs, ys, c=[constants_cmap.DATASETS_COLORS[constants_cmap.ALL_DATASET_NAMES.index(test_loader.dataset.y_unique_names[a])] for a in labels])

    colormap = cm.jet
    label_unique = np.arange(len(constants_cmap.DATASETS_NAMES))
    colorlist_unique = [ml_colors.rgb2hex(colormap(a)) for a in
                    label_unique / float(max(labels))]
    patches = [Line2D([0], [0], marker='o', color='gray', label=constants_cmap.ALL_DATASET_NAMES[a],
                      markerfacecolor=c) for a, c in zip(label_unique, colorlist_unique)]

    for a in label_unique:
        plt.scatter([np.median([xs[i] for i, b in enumerate(labels) if a==b])],[np.median([ys[i] for i, b in enumerate(labels) if a==b])], s=2000, c=colorlist_unique[a], cmap='jet', alpha=0.5)
        plt.annotate(constants_cmap.DATASETS_NAMES[a],
                     xy=(np.median([xs[i] for i, b in enumerate(labels) if a==b]), np.median([ys[i] for i, b in enumerate(labels) if a==b])), xytext=(-20, 20), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=3, headlength=2))

    plt.legend(handles=patches)
    plt.savefig(os.path.join(path_to_save, "latent_scatter{}.png".format(suffix)))


def main(model,use_z, fraction, epoch_checkpoint=300, suffix=""):


    n_latent_layer=2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    genes=None
    genes_name=None
    # genes=np.load("/media/hag007/Data/dlproj/cache_global/datasets/vemurafenib_resveratrol_olaparib/genes.npy", allow_pickle=True)
    # genes_name="vemurafenib_resveratrol_olaparib"

    dataset_names=constants_cmap.ALL_DATASET_NAMES # [a for i, a in enumerate(constants.DATASETS_NAMES) if constants.DATASETS_DICT[a]]
    dataset= datasets.Dataset(dataset_names, constants_cmap.DATA_TYPE)
    dataloader_ctor= datasets.DataLoader(dataset, 0.2, 0.2)
    testloader = dataloader_ctor.test_loader()

    dataloader_ctor_mask= datasets.DataLoader(dataset, 0.2, 0, 2)
    trainloader = dataloader_ctor_mask.train_loader()
    validationloader = dataloader_ctor_mask.valid_loader()

    encoder=Encoder(n_latent_layer=n_latent_layer)
    decoder=Decoder(n_latent_layer=n_latent_layer)
    classifier=Classifier(n_input_layer=n_latent_layer, n_classes=(len(constants_cmap.DATASETS_FILES) if genes_name is None else genes_name.count("_") + 1))

    path_format_to_save=os.path.join(constants_cmap.CACHE_GLOBAL_DIR, constants_cmap.DATA_TYPE, "model_{}_{}_{}_{{}}".format(fraction, model, "z" if use_z else "mu"))
    PATH_ENCODER= os.path.join(path_format_to_save,"ENC_mdl")
    PATH_DECODER= os.path.join(path_format_to_save,"DEC_mdl")
    if model!=constants_cmap.MODEL_VAE:
        PATH_CLASSIFIER= os.path.join(path_format_to_save,"CLS_mdl")

    load_model=True
    if load_model and os.path.exists(PATH_ENCODER.format(epoch_checkpoint)+suffix):
        encoder.load_state_dict(torch.load(PATH_ENCODER.format(epoch_checkpoint)+suffix))
        decoder.load_state_dict(torch.load(PATH_DECODER.format(epoch_checkpoint)+suffix))
        encoder.eval()
        decoder.eval()

        if model!=constants_cmap.MODEL_VAE:
            classifier.load_state_dict(torch.load(PATH_CLASSIFIER.format(epoch_checkpoint)+suffix))
            classifier.eval()

    with torch.no_grad():
        extract_latent_dimension(encoder if model != constants_cmap.MODEL_CLS else torch.nn.Sequential(encoder, classifier), trainloader, device, suffix + "_train", path_format_to_save.format(epoch_checkpoint))
        extract_latent_dimension(encoder if model != constants_cmap.MODEL_CLS else torch.nn.Sequential(encoder, classifier), validationloader, device, suffix + "_validation", path_format_to_save.format(epoch_checkpoint))
        extract_latent_dimension(encoder if model != constants_cmap.MODEL_CLS else torch.nn.Sequential(encoder, classifier), testloader, device, suffix + "_test", path_format_to_save.format(epoch_checkpoint))


if __name__=="__main__":


    fractions= [0.125] # [0.0, 0.01,0.03,0.05, 0.0625, 0.1, 0.2, 0.33, 0.5, 0.67] #[0.0, 0.01,0.03,0.05, 0.1, 0.2]# ], 0.1, 0.2] # [0.0, 0.005, 0.01, 0.02,0.03,0.04,0.05, 0.1, 0.2, 0.33, 0.5, 0.67, 1.0] # [0.03, 0.04] # , 0.1, 0.33, 0.67, 1.0]
    use_zs=[True] # , False]
    models=  [constants_cmap.MODEL_CLS, constants_cmap.MODEL_FULL] # [constants.MODEL_FULL, constants.MODEL_CLS, constants.MODEL_VAE]
    epoch_checkpoints=[300] # [100,300, 500, 1000, 1500, 2000] # [50, 100, 150,200,250,300]
    params=[]

    suffix= ""
    for cur_use_z in use_zs:
        for cur_fraction in fractions:
            for model in models:
                for cur_epoch_checkpoint in epoch_checkpoints:
                    print("start {} {} use_z={}".format(cur_fraction, model, cur_use_z))
                    # params.append([main, [model, cur_use_z, cur_fraction, cur_epoch_checkpoint, "_min"]])
                    main(model=model, use_z=cur_use_z, fraction=cur_fraction, epoch_checkpoint=cur_epoch_checkpoint, suffix=suffix)

    # p.map(func_star, params)
