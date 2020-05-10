import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import constants_tcga as constants
import os
df=pd.read_csv(os.path.join(constants.OUTPUT_GLOBAL_DIR, "KNN_fractions.csv"), sep='\t', index_col=0)


plt.cla()
originals=df.iloc[0,:]
fractions=df.iloc[1:,:]

plt.plot([0.01, 0.95], [originals.loc["PCA_CMAP"],originals.loc["PCA_CMAP"]], linestyle=(0,(5,10)), label="knn accuracy of original labels trained on PCA in CMAP")
plt.plot([0.01, 0.95], [originals.loc["VAE_CMAP"],originals.loc["VAE_CMAP"]], linestyle=(2,(5,10)) , label="knn accuracy of original labels trained on VAE in CMAP")
plt.plot(fractions.index, fractions.loc[:,"PCA_CMAP"], label="knn accuracy of new labels trained on PCA in CMAP")
plt.plot(fractions.index, fractions.loc[:,"VAE_CMAP"] , label="knn accuracy of new labels trained on VAE in CMAP")
plt.xlabel("test fraction")
plt.ylabel("knn score")
plt.legend()
plt.legend(loc='lower left')
plt.savefig(os.path.join(constants.OUTPUT_GLOBAL_DIR, "new_label_plot_CMAP.png"))

plt.cla()
plt.plot([0.01, 0.95], [originals.loc["PCA_TCGA"],originals.loc["PCA_TCGA"]], linestyle=(0,(5,10)), label="knn accuracy of original labels trained on PCA in TCGA")
plt.plot([0.01, 0.95], [originals.loc["VAE_TCGA"],originals.loc["VAE_TCGA"]], linestyle=(2,(5,10)), label="knn accuracy of original labels trained on VAE in TCGA")
plt.plot(fractions.index, fractions.loc[:,"PCA_TCGA"], label="knn accuracy of new labels trained on PCA in TCGA")
plt.plot(fractions.index, fractions.loc[:,"VAE_TCGA"] , label="knn accuracy of new labels trained on VAE in TCGA")
plt.xlabel("test fraction")
plt.ylabel("knn score")
plt.legend(loc='lower left')
plt.savefig(os.path.join(constants.OUTPUT_GLOBAL_DIR, "new_label_plot_TCGA.png"))