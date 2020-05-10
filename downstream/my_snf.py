import snf
import pandas as pd
import os
import constants_cmap
from sklearn.cluster import spectral_clustering, k_means
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
pts=[os.path.join(constants_cmap.OUTPUT_GLOBAL_DIR, pt) for pt in os.listdir(constants_cmap.OUTPUT_GLOBAL_DIR) if pt.startswith("diff_1000_min_tcga_")]

dfs=[]
dms=[]
for pt in pts:
    dfs.append(pd.read_csv(pt,index_col=0,sep='\t').T.sort_index())
    dms.append(distance_matrix(dfs[-1], dfs[-1]))

print(len(dfs))

max_d=-1
for dm in dms:
    max_d=max(dm.max(),max_d)

# for i, dm in enumerate(dms):
#     dms[i]= max_d-dm
    # sns.heatmap(pd.DataFrame(data=dms[i], index=dfs[0].index))
    # plt.savefig(os.path.join(constants.OUTPUT_GLOBAL_DIR, "hc.png"))
    # plt.clf()

df_concat = np.array(dms)
dm_avg=np.average(df_concat,axis=0)

sns.heatmap(pd.DataFrame(data=dm_avg, index=dfs[0].index))
plt.savefig(os.path.join(constants_cmap.OUTPUT_GLOBAL_DIR, "hc1.png"))
plt.clf()

sns.clustermap(pd.DataFrame(data=dm_avg, index=dfs[0].index, columns=dfs[0].index), method='single')
plt.savefig(os.path.join(constants_cmap.OUTPUT_GLOBAL_DIR, "hc2.png"))
plt.clf()

#
# affinity_networks = snf.make_affinity(dfs, metric='euclidean', K=5,  normalize=True, mu=1.0)
# fused_network = snf.snf([dms[0],dms[0]], K=5)
# best, second = snf.get_n_clusters(fused_network)
# labels_1 = spectral_clustering(fused_network, n_clusters=7)
# labels_2 = k_means(fused_network, n_clusters=7)
#
# sns.clustermap(pd.DataFrame(data=fused_network, index=dfs[0].index))
# plt.savefig(os.path.join(constaglents.OUTPUT_GLOBAL_DIR, "hc3.png"))
# print(pd.DataFrame(data=np.array([labels_1, labels_2[1]]).T, index=dfs[0].index))