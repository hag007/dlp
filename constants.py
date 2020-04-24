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

# DATASETS_FILES=["profiles_bortezomib_6290_level_4_GSE92742_GSE70138.tsv",
#                 "profiles_EMPTY_VECTOR_5114_level_4_GSE92742_GSE70138.tsv"] #
# ["profiles_sorafenib_1002_level_4_GSE92742_GSE70138.tsv",
# "profiles_saracatinib_1175_level_4_GSE92742_GSE70138.tsv"]#,
# "profiles_selumetinib_1008_level_4_GSE92742_GSE70138.tsv",
DATASETS_FILES=       [
    # "profiles_DMSO_46428_level_4_GSE92742_GSE70138.tsv",
    # "profiles_UnTrt_18246_level_4_GSE92742_GSE70138.tsv",
    # "profiles_GFP_6656_level_4_GSE92742_GSE70138.tsv",
    # "profiles_bortezomib_6290_level_4_GSE92742_GSE70138.tsv",
    # "profiles_MG-132_6289_level_4_GSE92742_GSE70138.tsv",
    # "profiles_lacZ_6014_level_4_GSE92742_GSE70138.tsv",
    # "profiles_LUCIFERASE_5761_level_4_GSE92742_GSE70138.tsv",
    # "profiles_pgw_5521_level_4_GSE92742_GSE70138.tsv",
    # "profiles_EMPTY_VECTOR_5114_level_4_GSE92742_GSE70138.tsv",
    # "profiles_vorinostat_4768_level_4_GSE92742_GSE70138.tsv",
    # "profiles_geldanamycin_3907_level_4_GSE92742_GSE70138.tsv",
    # "profiles_trichostatin-a_3692_level_4_GSE92742_GSE70138.tsv",
    # "profiles_wortmannin_3263_level_4_GSE92742_GSE70138.tsv",
    # "profiles_RFP_3171_level_4_GSE92742_GSE70138.tsv",
    # "profiles_ERG_2499_level_4_GSE92742_GSE70138.tsv",
    # "profiles_GSK-1059615_2369_level_4_GSE92742_GSE70138.tsv",
    # "profiles_PD-0325901_2320_level_4_GSE92742_GSE70138.tsv",
    # "profiles_sirolimus_2220_level_4_GSE92742_GSE70138.tsv",
    # "profiles_withaferin-a_1616_level_4_GSE92742_GSE70138.tsv",
    # "profiles_CGP-60474_1533_level_4_GSE92742_GSE70138.tsv",
    # "profiles_tozasertib_1409_level_4_GSE92742_GSE70138.tsv",
    # "profiles_BRAF_1372_level_4_GSE92742_GSE70138.tsv",
    # "profiles_sulforaphane_1320_level_4_GSE92742_GSE70138.tsv",
    # "profiles_ERBB3_1372_level_4_GSE92742_GSE70138.tsv",
    # "profiles_fulvestrant_1320_level_4_GSE92742_GSE70138.tsv",
    # "profiles_KRAS_1326_level_4_GSE92742_GSE70138.tsv",
    # "profiles_genistein_1314_level_4_GSE92742_GSE70138.tsv",
    # "profiles_mitoxantrone_1328_level_4_GSE92742_GSE70138.tsv",
    # "profiles_JUN_1286_level_4_GSE92742_GSE70138.tsv",
    # "profiles_EGFR_1268_level_4_GSE92742_GSE70138.tsv",
    # "profiles_estradiol_1263_level_4_GSE92742_GSE70138.tsv",
    # "profiles_veliparib_1259_level_4_GSE92742_GSE70138.tsv",
    # "profiles_BI-2536_1255_level_4_GSE92742_GSE70138.tsv",
    # "profiles_HcRed_1176_level_4_GSE92742_GSE70138.tsv",
    # "profiles_saracatinib_1175_level_4_GSE92742_GSE70138.tsv",
    # "profiles_tamoxifen_1133_level_4_GSE92742_GSE70138.tsv",
    # "profiles_radicicol_1133_level_4_GSE92742_GSE70138.tsv",
    # "profiles_resveratrol_1121_level_4_GSE92742_GSE70138.tsv",
    # "profiles_LY-294002_1124_level_4_GSE92742_GSE70138.tsv",
    # "profiles_vemurafenib_1064_level_4_GSE92742_GSE70138.tsv",
    # "profiles_eGFP_1063_level_4_GSE92742_GSE70138.tsv",
    # "profiles_BRD-K98948170_1049_level_4_GSE92742_GSE70138.tsv",
    # "profiles_selumetinib_1008_level_4_GSE92742_GSE70138.tsv",
    # "profiles_sorafenib_1002_level_4_GSE92742_GSE70138.tsv",




    # #################### bit sets ################################
    # "profiles_bortezomib_6290_level_4_GSE92742_GSE70138.tsv",
    # "profiles_lacZ_6014_level_4_GSE92742_GSE70138.tsv",
    # "profiles_LUCIFERASE_5761_level_4_GSE92742_GSE70138.tsv",
    # "profiles_EMPTY_VECTOR_5114_level_4_GSE92742_GSE70138.tsv",







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

DATASETS_NAMES=[a.split("_")[1] for a in DATASETS_FILES]

CANCER_TYPES= ["LUSC", "KIRC" , "KIRP", "LUSC", "LUAD", "COAD", "BRCA", "STAD", "LIHC", "READ", "PRAD", "BLCA", "HNSC", "THCA", "UCEC", "OV", "PAAD"]
# DATASETS_FILES=[
#     # "BLCA_normal_19.tsv",
#     # "BLCA_tumor_411.tsv",
#     "BRCA_normal_113.tsv",
#     "BRCA_tumor_1097.tsv",
#     "COAD_normal_41.tsv",
#     "COAD_tumor_469.tsv",
#     # "BLCA_normal_19.tsv",
#     # "BLCA_tumor_411.tsv",
#     # "COAD_tumor_469.tsv",
#     # "HNSC_normal_44.tsv",
#     # "HNSC_tumor_500.tsv",
#     "KIRC_normal_72.tsv",
#     "KIRC_tumor_534.tsv",
#     "KIRP_normal_32.tsv",
#     "KIRP_tumor_288.tsv",
#     "LIHC_normal_50.tsv",
#     "LIHC_tumor_371.tsv",
#     "LUAD_normal_59.tsv",
#     "LUAD_tumor_524.tsv",
#     "LUSC_normal_49.tsv",
#     "LUSC_tumor_501.tsv",
#     # "OV_normal_0.tsv",
#     # "OV_tumor_374.tsv",
#     # "PAAD_normal_4.tsv",
#     "PAAD_tumor_177.tsv",
#     "PRAD_normal_52.tsv",
#     # "PRAD_tumor_498.tsv",
#     # "READ_normal_10.tsv",
#     # "READ_tumor_166.tsv",
#     # "STAD_normal_32.tsv",
#     # "STAD_tumor_375.tsv",
#     "THCA_normal_58.tsv",
#     "THCA_tumor_502.tsv",
#     # "UCEC_normal_35.tsv",
#     # "UCEC_tumor_547.tsv"
#
# ]
#
# DATASETS_NAMES=["_".join(a.split("_")[0:2]) for a in DATASETS_FILES]
DATASETS_INCLUDED= [True for a in DATASETS_FILES]
DATASETS_F=[0 for a in DATASETS_FILES]
DATASETS_F_NAMES=["nib" for a in DATASETS_FILES]



DATASETS_COLORS=sns.color_palette("Paired", n_colors=len(DATASETS_FILES))

PATCHES = [Line2D([0], [0], marker='o', color='gray', label=DATASETS_NAMES[i], markersize=12, markerfacecolor=DATASETS_COLORS[i], alpha=0.7) for i, a in enumerate(DATASETS_NAMES) if DATASETS_INCLUDED[i]]

