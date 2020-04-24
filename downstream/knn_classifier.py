from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import constants
import matplotlib.pyplot as plt
import seaborn as sns

def main(path_to_save, model, cur_epoch, fraction, suffix, ax):

    score=knn(path_to_save, suffix)
    ax.scatter(cur_epoch, score, c=sns.color_palette("Paired", n_colors=len(models))[models.index(model)], label=("{}_{}".format(model,fraction) if cur_epoch==100 else None))


def knn(path_to_save, suffix):
    ds_train = np.load(os.path.join(path_to_save, "latent_features{}_train.npy".format(suffix)))
    X_train = ds_train[:, :-2].astype(np.float)
    y_train = ds_train[:, -2].astype(np.int)
    ds_test = np.load(os.path.join(path_to_save, "latent_features{}_test.npy".format(suffix)))
    X_test = ds_test[:, :-2].astype(np.float)
    y_test = ds_test[:, -2].astype(np.int)
    neigh = KNeighborsClassifier(n_neighbors=50)
    neigh.fit(X_train, y_train)
    score=neigh.score(X_test, y_test)
    return score



if __name__=="__main__":

    filter_func_dict={
        0.01:lambda a: a % 100 == 0,
        0.02:lambda a: a % 50 == 0,
        0.03:lambda a: a % 33 == 0,
        0.04:lambda a: a % 25 == 0,
        0.05:lambda a: a % 20 == 0,
        0.1:lambda a: a % 10 == 0,
        0.33:lambda a: a % 3 == 0,
        0.67:lambda a: a % 3 > 0,
        1.0:lambda a: True
    }

    fractions=[0.03, 0.04] # , 0.33, 0.67, 1.0]
    use_zs=[True] # , False]
    models=[constants.MODEL_FULL, constants.MODEL_VAE, constants.MODEL_CLS]

    epoch_checkpoints=[100,300, 500, 1000, 1500, 2000] # [50, 100, 150,200,250,300]


    suffix= ""# "_min"
    for cur_use_z in use_zs:
        for cur_fraction in fractions:
            fig,ax=plt.subplots(figsize=(10,10))
            for model in models:
                for cur_epoch in epoch_checkpoints:
                    path_format_to_save=os.path.join(constants.CACHE_GLOBAL_DIR, constants.DATA_TYPE, "model_{}_{}_{}_{}".format(cur_fraction,model,("z" if cur_use_z else "mu"), cur_epoch))
                    main(path_format_to_save, model, cur_epoch, cur_fraction, suffix, ax)

            ax.legend()
            plt.savefig(os.path.join(constants.OUTPUT_GLOBAL_DIR, "knn_score_{}.png".format(cur_fraction)))
            plt.clf()