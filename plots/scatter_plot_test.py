import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch.nn import functional as F
from torch import tensor

import constants_tcga as constants
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.cm as cm
import matplotlib.colors as ml_colors
from matplotlib.lines import Line2D

def plot(model, test_loader, device, suffix, dataset_names, path_to_save):
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

    X_pca = zs.detach().cpu().numpy() # PCA(n_components=2).fit_transform(zs.detach().cpu().numpy())
    xs,ys=list(zip(*X_pca))
    plt.subplots(figsize=(20,20))
    labels=labels.cpu().numpy()
    plt.scatter(xs,ys, c=[a for a in labels], cmap='jet')

    colormap = cm.jet
    label_unique = np.arange(len(dataset_names))
    colorlist_unique = [ml_colors.rgb2hex(colormap(a)) for a in
                    label_unique / float(max(labels))]
    patches = [Line2D([0], [0], marker='o', color='gray', label=dataset_names[a],
                  markerfacecolor=c) for a, c in zip(label_unique, colorlist_unique)]

    for a in label_unique:
        plt.scatter([np.median([xs[i] for i, b in enumerate(labels) if a==b])],[np.median([ys[i] for i, b in enumerate(labels) if a==b])], s=2000, c=colorlist_unique[a], cmap='jet', alpha=0.5)
        plt.annotate(dataset_names[a],
                    xy=(np.median([xs[i] for i, b in enumerate(labels) if a==b]), np.median([ys[i] for i, b in enumerate(labels) if a==b])), xytext=(-20, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=3, headlength=2))
    plt.legend(handles=patches)
    plt.savefig(os.path.join(path_to_save, "zs_scatter{}.png".format(suffix)))

