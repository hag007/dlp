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
DATA_TYPE="cmap"
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


DATASETS_FILES=       [

    ######################## histone deacetylation ###############3
    "profiles_trichostatin-a_3692_level_4_GSE92742_GSE70138.tsv",
    "profiles_vorinostat_4768_level_4_GSE92742_GSE70138.tsv",
    ####################### HSP #########################
    "profiles_geldanamycin_3907_level_4_GSE92742_GSE70138.tsv",
    "profiles_radicicol_1133_level_4_GSE92742_GSE70138.tsv",
    ############## breast cancer (hormones)#####################
    # "profiles_tamoxifen_1133_level_4_GSE92742_GSE70138.tsv",
    # "profiles_fulvestrant_1320_level_4_GSE92742_GSE70138.tsv",
    ################## breast cancer PARP #########################
    "profiles_veliparib_1259_level_4_GSE92742_GSE70138.tsv",
    "profiles_olaparib_1445_level_4_GSE92742_GSE70138.tsv",
    ############## hormones ###################
    # "profiles_estradiol_1263_level_4_GSE92742_GSE70138.tsv",
    ############## MAPK ##########################################
    # "profiles_wortmannin_3263_level_4_GSE92742_GSE70138.tsv",
    # "profiles_selumetinib_1008_level_4_GSE92742_GSE70138.tsv",
    ###################### others #################################
    "profiles_sirolimus_2220_level_4_GSE92742_GSE70138.tsv",
    # "profiles_sulforaphane_1320_level_4_GSE92742_GSE70138.tsv",
    # "profiles_withaferin-a_1616_level_4_GSE92742_GSE70138.tsv",


]

DATASETS_FILES=       [


    ######################## histone deacetylation ###############3
    "profiles_trichostatin-a_3692_level_4_GSE92742_GSE70138.tsv",
    # "profiles_vorinostat_4768_level_4_GSE92742_GSE70138.tsv",
    ####################### HSP #########################
    "profiles_geldanamycin_3907_level_4_GSE92742_GSE70138.tsv",
    # "profiles_radicicol_1133_level_4_GSE92742_GSE70138.tsv",
    ############## breast cancer (hormones)#####################
    # "profiles_tamoxifen_1133_level_4_GSE92742_GSE70138.tsv",
    # "profiles_fulvestrant_1320_level_4_GSE92742_GSE70138.tsv",
    ################## breast cancer PARP #########################
    "profiles_veliparib_1259_level_4_GSE92742_GSE70138.tsv",
    # "profiles_olaparib_1445_level_4_GSE92742_GSE70138.tsv",
    ############## hormones ###################
    # "profiles_estradiol_1263_level_4_GSE92742_GSE70138.tsv",
    ############## MAPK ##########################################
    # "profiles_wortmannin_3263_level_4_GSE92742_GSE70138.tsv",
    # "profiles_selumetinib_1008_level_4_GSE92742_GSE70138.tsv",
    ###################### others #################################
    # "profiles_sirolimus_2220_level_4_GSE92742_GSE70138.tsv",
    # "profiles_sulforaphane_1320_level_4_GSE92742_GSE70138.tsv",
    # "profiles_withaferin-a_1616_level_4_GSE92742_GSE70138.tsv",


]

NEW_DATASETS_FILES=       [

    ######################## histone deacetylation ###############3
    # "profiles_trichostatin-a_3692_level_4_GSE92742_GSE70138.tsv",
    "profiles_vorinostat_4768_level_4_GSE92742_GSE70138.tsv",
    ####################### HSP #########################
    # "profiles_geldanamycin_3907_level_4_GSE92742_GSE70138.tsv",
    "profiles_radicicol_1133_level_4_GSE92742_GSE70138.tsv",
    ############## breast cancer (hormones)#####################
    # "profiles_tamoxifen_1133_level_4_GSE92742_GSE70138.tsv",
    # "profiles_fulvestrant_1320_level_4_GSE92742_GSE70138.tsv",
    ################## breast cancer PARP #########################
    # "profiles_veliparib_1259_level_4_GSE92742_GSE70138.tsv",
    "profiles_olaparib_1445_level_4_GSE92742_GSE70138.tsv",
    ############## hormones ###################
    # "profiles_estradiol_1263_level_4_GSE92742_GSE70138.tsv",
    ############## MAPK ##########################################
    # "profiles_wortmannin_3263_level_4_GSE92742_GSE70138.tsv",
    # "profiles_selumetinib_1008_level_4_GSE92742_GSE70138.tsv",
    ###################### others #################################
    # "profiles_sirolimus_2220_level_4_GSE92742_GSE70138.tsv",
    # "profiles_sulforaphane_1320_level_4_GSE92742_GSE70138.tsv",
    # "profiles_withaferin-a_1616_level_4_GSE92742_GSE70138.tsv",


]

NEW_DATASETS_NAMES=[a.split("_")[1] for a in NEW_DATASETS_FILES ]#if not DATASETS_DICT["_".join(a.split("_")[0:2])]]

DATASETS_NAMES=[a.split("_")[1] for a in DATASETS_FILES]
ALL_DATASET_NAMES=DATASETS_NAMES
DATASETS_COLORS=sns.color_palette("Paired", n_colors=len(DATASETS_FILES))
PATCHES = [Line2D([0], [0], marker='o', color='gray', label=DATASETS_NAMES[i], markersize=12, markerfacecolor=DATASETS_COLORS[i], alpha=0.7) for i, a in enumerate(DATASETS_NAMES)]

