from nn.tcga.run_tcga import main as main_run
from downstream.tcga.extract_latent_features import main as main_extraction
from downstream.cancer_normal_distance import main as main_distance
import constants_tcga as constants
import numpy as np


cp=1000
for a in np.arange(10000):
    main_run(model=constants.MODEL_VAE, use_z=True, fraction=1.0, max_epoch=cp, epoch_checkpoint=0)
    main_extraction(model=constants.MODEL_VAE, use_z=True, fraction=1.0, epoch_checkpoint=cp, suffix="_min")
    main_distance(model=constants.MODEL_VAE, use_z=True, fraction=1.0, epoch_checkpoint=cp, suffix="_min")

    main_extraction(model=constants.MODEL_VAE, use_z=True, fraction=1.0, epoch_checkpoint=100, suffix="")
    main_distance(model=constants.MODEL_VAE, use_z=True, fraction=1.0, epoch_checkpoint=100, suffix="")

    main_extraction(model=constants.MODEL_VAE, use_z=True, fraction=1.0, epoch_checkpoint=500, suffix="")
    main_distance(model=constants.MODEL_VAE, use_z=True, fraction=1.0, epoch_checkpoint=500, suffix="")

    main_extraction(model=constants.MODEL_VAE, use_z=True, fraction=1.0, epoch_checkpoint=1000, suffix="")
    main_distance(model=constants.MODEL_VAE, use_z=True, fraction=1.0, epoch_checkpoint=1000, suffix="")
