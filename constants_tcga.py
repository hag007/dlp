import simplejson as json
import os
import seaborn as sns
sns.set(color_codes=True)
from matplotlib.lines import Line2D

dir_path = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONF = "config/conf.json"
config_json = json.load(open(os.path.join(dir_path, PATH_TO_CONF)))

REPO_DIR = os.path.dirname(os.path.realpath(__file__))
SH_DIR = os.path.join(REPO_DIR, "sh","scripts")


MODEL_VAE="vae"
MODEL_CLS="cls"
MODEL_FULL="full"

SERVER_MODE = False
REPORTS = True
DISEASE_MODE = False
HG_MODE = True
ALGO_HG_MODE=False
EMB_MODE = True
USE_CACHE =True # True
PHENOTYPE_FORMAT = "GDC-TCGA"#"GDC"
DATASET_TYPE = "GDC"
DATA_TYPE="tcga"
CANCER_TYPE = "SKCM"
BASE_PROFILE= config_json['BASE_PROFILE']
DATASET_NAME = "TNFa_2"
DATASETS_DIR= os.path.join(BASE_PROFILE, "cmap_datasets")
NETWORKS_DIR = os.path.join(BASE_PROFILE, "networks")
TEMPLATES_DIR = os.path.join(BASE_PROFILE, "templates")
DATASET_DIR = os.path.join(DATASETS_DIR, DATASET_NAME)
DATA_DIR = os.path.join(DATASET_DIR, "data")
CACHE_DIR = os.path.join(DATASET_DIR, "cache")
DICTIONARIES_DIR = os.path.join(BASE_PROFILE, "dictionaries")
OUTPUT_DIR = os.path.join(DATASET_DIR, "output")
OUTPUT_GLOBAL_DIR = os.path.join(BASE_PROFILE, "output")
TCGA_DATA_DIR = os.path.join(DATASET_DIR, "data")
GO_DIR = os.path.join(BASE_PROFILE, "GO")
CACHE_GLOBAL_DIR = os.path.join(BASE_PROFILE, "cache_global")
LIST_DIR = os.path.join(BASE_PROFILE, "list")
REPOS_DIR = os.path.join(BASE_PROFILE, "repos")
RAW_DIR = os.path.join(BASE_PROFILE, "raw")

# CANCER_TYPES= ["LUSC", "KIRC" , "KIRP", "LUSC", "LUAD", "COAD", "BRCA", "STAD", "LIHC", "READ", "PRAD", "BLCA", "HNSC", "THCA", "UCEC", "OV", "PAAD"]
DATASETS_FILES=[
    "BLCA_normal_19.tsv",
    "BLCA_tumor_411.tsv",
    "BRCA_normal_113.tsv",
    "BRCA_tumor_1097.tsv",
    "COAD_normal_41.tsv",
    "COAD_tumor_469.tsv",
    "HNSC_normal_44.tsv",
    "HNSC_tumor_500.tsv",
    "KIRC_normal_72.tsv",
    "KIRC_tumor_534.tsv",
    "KIRP_normal_32.tsv",
    "KIRP_tumor_288.tsv",
    "LIHC_normal_50.tsv",
    "LIHC_tumor_371.tsv",
    "LUAD_normal_59.tsv",
    "LUAD_tumor_524.tsv",
    "LUSC_normal_49.tsv",
    "LUSC_tumor_501.tsv",
    # "OV_normal_0.tsv",
    "OV_tumor_374.tsv",
    "PAAD_normal_4.tsv",
    "PAAD_tumor_177.tsv",
    "PRAD_normal_52.tsv",
    "PRAD_tumor_498.tsv",
    "READ_normal_10.tsv",
    "READ_tumor_166.tsv",
    "STAD_normal_32.tsv",
    "STAD_tumor_375.tsv",
    # "THCA_normal_58.tsv",
    # "THCA_tumor_502.tsv",
    # "UCEC_normal_35.tsv",
    # "UCEC_tumor_547.tsv"

]

DATASETS_FILES=[
    # "BLCA_normal_19.tsv",
    "BLCA_tumor_411.tsv",
    # "BRCA_normal_113.tsv",
    "BRCA_tumor_1097.tsv",
    # "COAD_normal_41.tsv",
    "COAD_tumor_469.tsv",
    # "HNSC_normal_44.tsv",
    "HNSC_tumor_500.tsv",
    # "KIRC_normal_72.tsv",
    # "KIRC_tumor_534.tsv",
    # "KIRP_normal_32.tsv",
    "KIRP_tumor_288.tsv",
    # "LIHC_normal_50.tsv",
    # "LIHC_tumor_371.tsv",
    # "LUAD_normal_59.tsv",
    "LUAD_tumor_524.tsv",
    # "LUSC_normal_49.tsv",
    # "LUSC_tumor_501.tsv",
    # "OV_normal_0.tsv",
    "OV_tumor_374.tsv",
    # "PAAD_normal_4.tsv",
    "PAAD_tumor_177.tsv",
    # "PRAD_normal_52.tsv",
    "PRAD_tumor_498.tsv",
    # "READ_normal_10.tsv",
    # "READ_tumor_166.tsv",
    # "STAD_normal_32.tsv",
    # "STAD_tumor_375.tsv",
    # "THCA_normal_58.tsv",
    # "THCA_tumor_502.tsv",
    # "UCEC_normal_35.tsv",
    # "UCEC_tumor_547.tsv"

]


NEW_DATASETS_FILES=[
    # "BLCA_normal_19.tsv",
    # "BLCA_tumor_411.tsv",
    # "BRCA_normal_113.tsv",
    # "BRCA_tumor_1097.tsv",
    # "COAD_normal_41.tsv",
    # "COAD_tumor_469.tsv",
    # "HNSC_normal_44.tsv",
    # "HNSC_tumor_500.tsv",
    # "KIRC_normal_72.tsv",
    "KIRC_tumor_534.tsv",
    # "KIRP_normal_32.tsv",
    # "KIRP_tumor_288.tsv",
    # "LIHC_normal_50.tsv",
    # "LIHC_tumor_371.tsv",
    # "LUAD_normal_59.tsv",
    # "LUAD_tumor_524.tsv",
    # "LUSC_normal_49.tsv",
    "LUSC_tumor_501.tsv",
    # "OV_normal_0.tsv",
    # "OV_tumor_374.tsv",
    # "PAAD_normal_4.tsv",
    # "PAAD_tumor_177.tsv",
    # "PRAD_normal_52.tsv",
    # "PRAD_tumor_498.tsv",
    # "READ_normal_10.tsv",
    "READ_tumor_166.tsv",
    # "STAD_normal_32.tsv",
    "STAD_tumor_375.tsv",
    # "THCA_normal_58.tsv",
    "THCA_tumor_502.tsv",
    # "UCEC_normal_35.tsv",
    "UCEC_tumor_547.tsv"

]

DATASETS_DICT={
    "BLCA_normal":  True,
    "BLCA_tumor":  True,
    "BRCA_normal":  True,
    "BRCA_tumor":  True,
    "COAD_normal":  True,
    "COAD_tumor":  True,
    "HNSC_normal":  True,
    "HNSC_tumor":  True,
    "KIRC_normal":  True,
    "KIRC_tumor":  True,
    "KIRP_normal":  True,
    "KIRP_tumor":  True,
    "LIHC_normal":  True,
    "LIHC_tumor":  True,
    "LUAD_normal":  True,
    "LUAD_tumor":  True,
    "LUSC_normal":  True,
    "LUSC_tumor":  True,
    # "OV_normal":  True,
    "OV_tumor":  True,
    "PAAD_normal":  True,
    "PAAD_tumor":  True,
    "PRAD_normal":  True,
    "PRAD_tumor":  True,
    "READ_normal":  True,
    "READ_tumor":  True,
    "STAD_normal":  True,
    "STAD_tumor":  True,
    "THCA_normal":  True,
    "THCA_tumor":  True,
    "UCEC_normal":  True,
    "UCEC_tumor":  True

}

ALL_DATASET_NAMES=["_".join(a.split("_")[0:2]) for a in DATASETS_FILES]
DATASETS_NAMES=["_".join(a.split("_")[0:2]) for a in DATASETS_FILES if DATASETS_DICT["_".join(a.split("_")[0:2])]]
DATASETS_F=[0 for a in DATASETS_FILES]
DATASETS_F_NAMES=["nib" for a in DATASETS_FILES]


ICGC_DATASETS_FILES=[
    "ICGC_BRCA_KR_50.tsv",
    # "ICGC_LICA_FR_161.tsv",
    # "ICGC_LIRI_JP_445.tsv",
    # "ICGC_RECA_EU_136.tsv",
    # "ICGC_PRAD_FR_25.tsv",
    # "ICGC_PAEN_AU_33.tsv",
    # "ICGC_PACA_AU_92.tsv",
    # "ICGC_OV_AU_111.tsv"
]

ICGC_DATASETS_DICT={
    "ICGC_BRCA": True,
    "ICGC_LICA": True,
    "ICGC_LIRI": True,
    "ICGC_PRAD": True,
    "ICGC_RECA": True,
    "ICGC_PAEN": True,
    "ICGC_PACA": True,
    "ICGC_OV": True,
}

ICGC_PSEUDO_LABELS={
    "ICGC_BRCA": 1,# 3,
    "ICGC_LICA": 5, # 13
    "ICGC_LIRI": 5,# 13
    "ICGC_PRAD": 10,# 22
    "ICGC_RECA": 4,# 9
    "ICGC_PAEN": 9,# 20
    "ICGC_PACA": 9,# 20
    "ICGC_OV": 8# 18
}

ICGC_DATASETS_NAMES=["_".join(a.split("_")[0:2]) for a in ICGC_DATASETS_FILES if ICGC_DATASETS_DICT["_".join(a.split("_")[0:2])]]
ICGC_ALL_DATASET_NAMES=["_".join(a.split("_")[0:2]) for a in ICGC_DATASETS_FILES]


# NEW_DATASETS_FILES=[
#
#     "KIRP_normal_32.tsv",
#     "KIRP_tumor_288.tsv",
#     "LUSC_normal_49.tsv",
#     "LUSC_tumor_501.tsv",
#     "OV_normal_0.tsv",
#     "READ_normal_10.tsv",
#     "READ_tumor_166.tsv",
#     "STAD_normal_32.tsv",
#     "STAD_tumor_375.tsv",
#     "UCEC_normal_35.tsv",
#     "UCEC_tumor_547.tsv"
#
# ]
#



NEW_DATASETS_NAMES=["_".join(a.split("_")[0:2]) for a in NEW_DATASETS_FILES ]#if not DATASETS_DICT["_".join(a.split("_")[0:2])]]


DATASETS_COLORS=sns.color_palette("Paired", n_colors=len(DATASETS_FILES))
PATCHES = [Line2D([0], [0], marker='o', color='gray', label=DATASETS_NAMES[i], markersize=12, markerfacecolor=DATASETS_COLORS[i], alpha=0.7) for i, a in enumerate(DATASETS_NAMES) if DATASETS_DICT[a]]

ICGC_DATASETS_COLORS=sns.color_palette("Set2", n_colors=len(ICGC_DATASETS_FILES))
ICGC_PATCHES = [Line2D([0], [0], marker='o', color='gray', label=ICGC_DATASETS_NAMES[i], markersize=12, markerfacecolor=ICGC_DATASETS_COLORS[i], alpha=0.7) for i, a in enumerate(ICGC_DATASETS_NAMES) if ICGC_DATASETS_DICT[a]]

# NEW_DATASETS_COLORS=sns.color_palette("Set2", n_colors=len(ICGC_DATASETS_FILES))
# NEW_PATCHES = [Line2D([0], [0], marker='o', color='gray', label=ICGC_DATASETS_NAMES[i], markersize=12, markerfacecolor=ICGC_DATASETS_COLORS[i], alpha=0.7) for i, a in enumerate(ICGC_DATASETS_NAMES)  if ICGC_DATASETS_DICT[a]]
