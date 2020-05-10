from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import constants_tcga as constants
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.cm as cm
import matplotlib.colors as ml_colors
from matplotlib.lines import Line2D

def main(path_to_save, model, cur_epoch, fraction, suffix, ax):

    score=knn(path_to_save, suffix)
    ax.scatter(cur_epoch, score, c=sns.color_palette("Paired", n_colors=len(models))[models.index(model)], label=("{}_{}".format(model,fraction) if cur_epoch==100 else None))
    return score


def knn(path_to_save, suffix):
    ds_train = np.load(os.path.join(path_to_save, "latent_features{}_train.npy".format(suffix)))
    X_train = ds_train[:, :-2].astype(np.float)
    y_train = ds_train[:, -2].astype(np.int)
    ds_test = np.load(os.path.join(path_to_save, "latent_features{}_test.npy".format(suffix)))
    X_test = ds_test[:, :-2].astype(np.float)
    y_test = ds_test[:, -2].astype(np.int)
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(X_train, y_train)
    score=neigh.score(X_test, y_test)
    print(score)
    print(neigh.score(X_train, y_train))

    dataset_names=constants.ALL_DATASET_NAMES
    plt.clf()
    fig,ax = plt.subplots(figsize=(20, 20))
    X=np.vstack([X_train,X_test])
    xs=X[:, 0]
    ys=X[:, 1]
    labels=np.hstack([y_train,y_test])
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
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow' , alpha=0.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=3, headlength=2))

    plt.legend(handles=patches)


    plt.savefig(os.path.join(constants.OUTPUT_GLOBAL_DIR, "clustering_vae_tcga_knn.png"))

    return score



if __name__=="__main__":

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

    fractions= [1.0] # [0.005, 0.01,0.03,0.05, 0.0625] # [0.0, 0.005, 0.01, 0.02,0.03,0.04,0.05, 0.1, 0.2, 0.33, 0.5, 0.67, 1.0] # [0.03, 0.04] # , 0.1, 0.33, 0.67, 1.0]
    use_zs=[True] # , False]
    models=  [constants.MODEL_VAE] # , constants.MODEL_CLS] # [constants.MODEL_FULL, constants.MODEL_CLS, constants.MODEL_VAE]
    epoch_checkpoints=[1000] # [100,300, 500, 1000, 1500, 2000] # [50, 100, 150,200,250,300]

    x_fractions={k: [] for k in models}
    y_scores=[]
    suffix=  "_min"
    for cur_use_z in use_zs:
        for cur_fraction in fractions:
            print("fraction: {}".format(cur_fraction))
            fig,ax=plt.subplots(figsize=(10,10))
            for model in models:
                for cur_epoch in epoch_checkpoints:
                    path_format_to_save=os.path.join(constants.CACHE_GLOBAL_DIR, constants.DATA_TYPE, "model_{}_{}_{}_{}".format(cur_fraction,model,("z" if cur_use_z else "mu"), cur_epoch))
                    y_scores.append(main(path_format_to_save, model, cur_epoch, cur_fraction, suffix, ax))
                    x_fractions[model].append(y_scores[-1])

            ax.legend()
            plt.savefig(os.path.join(constants.OUTPUT_GLOBAL_DIR, "knn_score_{}.png".format(cur_fraction)))
            plt.clf()


    fig,ax = plt.subplots(figsize=(10, 10))
    # plt.plot([0.01 ,0.33],[0.406 ,0.406], label="VAE")
    for k,v in x_fractions.items():
        plt.plot(fractions,v, label=("SSV" if k=="full" else "FCN"))
    plt.xlabel("fractions")
    plt.ylabel("knn accuracy")
    plt.legend()
    plt.savefig(os.path.join(constants.OUTPUT_GLOBAL_DIR,"knn_fraction_to_accuracy_{}_{}{}.png".format(model,cur_epoch,suffix)))
