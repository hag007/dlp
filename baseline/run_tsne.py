import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import constants
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datasets import cmap_datasets


def main():
    print("start tsne...")
    dataset= cmap_datasets.CMAPDataset()
    labels=dataset.y
    n_components=2
    X_pca = PCA(n_components=100).fit_transform(dataset.X)
    X = TSNE(n_components=n_components, metric="euclidean", perplexity=15.0).fit_transform(X_pca)
    fig = plt.figure(1, figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], c=labels)
    plt.savefig(os.path.join(constants.OUTPUT_GLOBAL_DIR, "tnse.png"))

if __name__=='__main__':
    main()