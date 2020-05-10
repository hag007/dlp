import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch import tensor

import constants_tcga as constants
import numpy as np
from multiprocessing import Pool
from sklearn.decomposition import PCA

import matplotlib.cm as cm
import matplotlib.colors as ml_colors
from matplotlib.lines import Line2D


def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return a_b[0](*a_b[1])



def plot(model, test_loader, device, suffix, path_to_save):
    zs=tensor([])
    mus=tensor([])
    logvars=tensor([])
    labels=tensor([]).long()

    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(device)
        z, mu, logvar = model(data)
        zs=torch.cat((zs, z), 0)
        mus=torch.cat((mus, mu), 0)
        logvars=torch.cat((logvars, logvar), 0)
        labels=torch.cat((labels, label), 0)

    # xs,ys=list(zip(*zs.cpu().numpy()))
    X_pca = PCA(n_components=2).fit_transform(zs.detach().cpu().numpy())
    xs,ys=list(zip(*X_pca))
    plt.subplots(figsize=(20,20))
    labels=labels.cpu().numpy()
    plt.scatter(xs,ys, c=[a for a in labels], cmap='jet') # sns.color_palette("Paired", n_colors=len(constants.DATASETS_INCLUDED))[a]


    colormap = cm.jet
    label_unique = np.arange(len(constants.DATASETS_NAMES))
    colorlist_unique = [ml_colors.rgb2hex(colormap(a)) for a in
                    label_unique / float(max(labels))]
    patches = [Line2D([0], [0], marker='o', color='gray', label=constants.DATASETS_NAMES[a],
                  markerfacecolor=c) for a, c in zip(label_unique, colorlist_unique)]

    for a in label_unique:
        plt.scatter([np.mean([xs[i] for i, b in enumerate(labels) if a==b])],[np.mean([ys[i] for i, b in enumerate(labels) if a==b])], s=2000, c=colorlist_unique[a], cmap='jet', alpha=0.5)
        plt.annotate(constants.DATASETS_NAMES[a],
                    xy=(np.mean([xs[i] for i, b in enumerate(labels) if a==b]), np.mean([ys[i] for i, b in enumerate(labels) if a==b])), xytext=(-20, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=3, headlength=2))
    plt.legend(handles=patches) # plt.legend(handles=[a for a in constants.PATCHES])
    plt.savefig(os.path.join(path_to_save, "zs_scatter{}.png".format(suffix)))



def main(model, use_z, fraction, max_epoch=300, epoch_checkpoint=0):



    path_to_save_format=os.path.join(constants.CACHE_GLOBAL_DIR, constants.DATA_TYPE, "model_{}_{}_{}_{{}}".format(fraction,model,"z" if use_z else "mu"))

    train_losses=[]
    valid_losses=[]
    if model!=constants.MODEL_FULL:
        train_file_name="train_losses.txt"
        val_file_name="val_losses.txt"
        fig,ax = plt.subplots()
        path_to_plot=os.path.join(constants.OUTPUT_GLOBAL_DIR, "losses_{}_{}_{}_{}_{}.png".format(constants.DATA_TYPE, fraction,model,("z" if use_z else "mu"), max_epoch))
        plot_losses(ax, epoch_checkpoint, max_epoch, path_to_plot, path_to_save_format, train_file_name, train_losses,
                val_file_name, valid_losses)
    else:
        fig,ax = plt.subplots()
        train_file_name="train_1_losses.txt"
        val_file_name="val_1_losses.txt"
        path_to_plot=os.path.join(constants.OUTPUT_GLOBAL_DIR, "losses_{}_{}_{}_{}_{}_1.png".format(constants.DATA_TYPE, fraction,model,("z" if use_z else "mu"), max_epoch))
        plot_losses(ax, epoch_checkpoint, max_epoch, path_to_plot, path_to_save_format, train_file_name, train_losses,
                val_file_name, valid_losses)

        fig,ax = plt.subplots()
        train_file_name="train_2_losses.txt"
        val_file_name="val_2_losses.txt"
        path_to_plot=os.path.join(constants.OUTPUT_GLOBAL_DIR, "losses_{}_{}_{}_{}_{}_2.png".format(constants.DATA_TYPE, fraction,model,("z" if use_z else "mu"), max_epoch))
        plot_losses(ax, epoch_checkpoint, max_epoch, path_to_plot, path_to_save_format, train_file_name, train_losses,
                val_file_name, valid_losses)


def plot_losses(ax, epoch_checkpoint, max_epoch, path_to_plot, path_to_save_format, train_file_name, train_losses,
                val_file_name, valid_losses):
    for cur_epoch in np.arange(epoch_checkpoint+50, max_epoch + 1,50):
        path_to_save = path_to_save_format.format(cur_epoch)

        train_losses = train_losses + [float(a.strip()) for a in
                                       open(os.path.join(path_to_save, train_file_name), 'r').readlines()]
        valid_losses = valid_losses + [float(a.strip()) for a in
                                       open(os.path.join(path_to_save, val_file_name), 'r').readlines()]
    ax_2 = ax.twinx()
    ax.plot(np.arange(epoch_checkpoint, max_epoch+1)[5:], train_losses[5:], label="train", color='blue')
    ax.plot([], [], label="valid", color='red')
    ax_2.plot(np.arange(epoch_checkpoint, max_epoch+1)[5:], valid_losses[5:], color='red')
    ax.set_xlabel("# of epoch")
    # ax.set_yscale("log")
    ax.set_ylabel("loss")
    ax.legend()
    # ax_2.legend()
    plt.savefig(path_to_plot)


if __name__=="__main__":
    filter_func_dict={
        0.01:lambda a: a % 100 == 0,
        0.03:lambda a: a % 50 == 0,
        0.05:lambda a: a % 20 == 0,
        0.1:lambda a: a % 10 == 0,
        0.2:lambda a: a % 5 == 0,
        0.5:lambda a: a % 2 == 0,
        1.0:lambda a: True
    }

    fractions=[0.1]
    use_zs=[True] # , False]
    models=  [constants.MODEL_FULL] # , constants.MODEL_CLS] # [constants.MODEL_FULL, constants.MODEL_CLS, constants.MODEL_VAE]

    p=Pool(2)
    params=[]
    max_epoch=1000
    epoch_checkpoint=0 # 300
    for cur_use_z in use_zs:
        for cur_fraction in fractions:
            for model in models:
                print("start {} {} use_z={}".format(cur_fraction, model, cur_use_z))
                # params.append([main, [model, cur_use_z, cur_fraction, max_epoch, epoch_checkpoint]])
                main(model=model, use_z=cur_use_z, fraction=cur_fraction, max_epoch=max_epoch, epoch_checkpoint=epoch_checkpoint)

    # p.map(func_star, params)
