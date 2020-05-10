import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch

from nn.models import Encoder, Decoder, Classifier
import constants_tcga as constants
from datasets import datasets
import torch.optim as optim
import numpy as np
import copy
from multiprocessing import Pool

from nn.tcga_ssv.flow_tcga_ssv import train_vae,train_cls,train_full
from plots.scatter_plot_test import plot


def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return a_b[0](*a_b[1])

def main(model, use_z, fraction, max_epoch=300, epoch_checkpoint=0):

    filter_func=filter_func_dict[fraction]

    n_latent_layer=2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    genes=None
    genes_name=None
    # genes=np.load("/media/hag007/Data/dlproj/cache_global/datasets/vemurafenib_resveratrol_olaparib/genes.npy", allow_pickle=True)
    # genes_name="vemurafenib_resveratrol_olaparib"

    dataset_names=constants.ALL_DATASET_NAMES
    dataset= datasets.Dataset(dataset_names=dataset_names, data_type=constants.DATA_TYPE)
    dataloader_ctor= datasets.DataLoader(dataset, 0.2, 0.2)
    testloader = dataloader_ctor.test_loader()

    dataset_mask= datasets.DatasetMask(dataset_names=dataset_names, data_type=constants.DATA_TYPE, filter_func=filter_func)
    dataloader_ctor_mask= datasets.DataLoader(dataset_mask, 0.2, 0, 2)
    trainloader = dataloader_ctor_mask.train_loader()
    validationloader = dataloader_ctor_mask.valid_loader()

    encoder=Encoder(n_latent_layer=n_latent_layer)
    decoder=Decoder(n_latent_layer=n_latent_layer)
    classifier=Classifier(n_input_layer=n_latent_layer, n_classes=(len(constants.DATASETS_FILES) if genes_name is None else genes_name.count("_") + 1)) #  * 2

    path_format_to_save=os.path.join(constants.CACHE_GLOBAL_DIR, constants.DATA_TYPE, "model_{}_{}_{}_{{}}".format(fraction,model,"z" if use_z else "mu"))
    PATH_ENCODER= os.path.join(path_format_to_save,"ENC_mdl")
    PATH_DECODER= os.path.join(path_format_to_save,"DEC_mdl")
    PATH_CLASSIFIER= os.path.join(path_format_to_save,"CLS_mdl")

    load_model=epoch_checkpoint >0
    if load_model and os.path.exists(PATH_ENCODER.format(epoch_checkpoint)):
        encoder.load_state_dict(torch.load(PATH_ENCODER.format(epoch_checkpoint)))
        encoder.eval()
        decoder.load_state_dict(torch.load(PATH_DECODER.format(epoch_checkpoint)))
        decoder.eval()
        classifier.load_state_dict(torch.load(PATH_CLASSIFIER.format(epoch_checkpoint)))
        classifier.eval()
    else:
        epoch_checkpoint=0


    lr_vae = 3e-4
    lr_cls = 3e-4
    parameters=list(encoder.parameters())+list(decoder.parameters())
    optimizer_vae = optim.Adam(parameters, lr=lr_vae)
    optimizer_cls = optim.Adam(list(encoder.parameters())+list(classifier.parameters()), lr=lr_cls)
    log_interval=100
    min_encoder=None
    min_decoder=None
    min_classifier=None
    min_epoch=-1
    min_val_loss=10e10
    train_losses=[]
    val_losses=[]
    for cur_epoch in np.arange(epoch_checkpoint, max_epoch + 1):
        if model==constants.MODEL_FULL:
            mdl=train_full
        elif model==constants.MODEL_CLS:
            mdl=train_cls
        elif model==constants.MODEL_VAE:
            mdl=train_vae
        else:
            raise

        factor_vae=1
        factor_cls=1
        train_loss, validation_loss, = mdl(cur_epoch, encoder, decoder, classifier, factor_vae, factor_cls, optimizer_vae, optimizer_cls, trainloader, validationloader, device, log_interval)
        train_losses.append(['{:.2f}'.format(a) for a in train_loss])
        val_losses.append(['{:.2f}'.format(a) for a in validation_loss])

        if min_val_loss>sum(validation_loss):
            min_encoder = copy.deepcopy(encoder)
            min_decoder = copy.deepcopy(decoder)
            min_classifier = copy.deepcopy(classifier)

            min_epoch=cur_epoch
            min_val_loss=sum(validation_loss)

        print("min_val_loss: {} (epoch n={})".format(min_epoch, min_val_loss))

        if (cur_epoch) % 50 == 0 and cur_epoch != epoch_checkpoint:
            try:
                os.makedirs(path_format_to_save.format(cur_epoch))
            except:
                pass
            if min_encoder is not None:
                torch.save(min_encoder.state_dict(), PATH_ENCODER.format(cur_epoch)+"_min")
                torch.save(min_decoder.state_dict(), PATH_DECODER.format(cur_epoch)+"_min")
                torch.save(min_classifier.state_dict(), PATH_CLASSIFIER.format(cur_epoch)+"_min")
                open(os.path.join(path_format_to_save.format(cur_epoch), "min_epoch.txt"), "w").write("{}_{}".format(min_val_loss,min_epoch))

                plot( min_encoder if model!=constants.MODEL_CLS or use_z else torch.nn.Sequential(min_encoder,min_classifier), testloader, device, "_min", dataset_names, path_format_to_save.format(cur_epoch))


            torch.save(encoder.state_dict(), PATH_ENCODER.format(cur_epoch))
            torch.save(decoder.state_dict(), PATH_DECODER.format(cur_epoch))
            torch.save(classifier.state_dict(), PATH_CLASSIFIER.format(cur_epoch))

            plot( encoder if model!=constants.MODEL_CLS or use_z else torch.nn.Sequential(encoder,classifier), testloader, device, "", dataset_names, path_format_to_save.format(cur_epoch))

            if model==constants.MODEL_FULL:
                open(os.path.join(path_format_to_save.format(cur_epoch), "train_1_losses.txt"), "w").write("\n".join([a[0] for a in train_losses]))
                open(os.path.join(path_format_to_save.format(cur_epoch), "train_2_losses.txt"), "w").write("\n".join([a[1] for a in train_losses]))
                open(os.path.join(path_format_to_save.format(cur_epoch), "val_1_losses.txt"), "w").write("\n".join([a[0] for a in val_losses]))
                open(os.path.join(path_format_to_save.format(cur_epoch), "val_2_losses.txt"), "w").write("\n".join([a[1] for a in val_losses]))
            else:
                open(os.path.join(path_format_to_save.format(cur_epoch), "train_losses.txt"), "w").write("\n".join([a[0] for a in train_losses]))
                open(os.path.join(path_format_to_save.format(cur_epoch), "val_losses.txt"), "w").write("\n".join([a[0] for a in val_losses]))
            train_losses=[]
            val_losses=[]


if __name__=="__main__":

    filter_func_dict={
        0.0:lambda a: False,
        0.005:lambda a: a % 200 == 0,
        0.01:lambda a: a % 100 == 0,
        0.02:lambda a: a % 50 == 0,
        0.03:lambda a: a % 33 ==0,
        0.04:lambda a: a % 25 == 0,
        0.05:lambda a: a % 20 == 0,
        0.0625:lambda a: a % 16 == 0,
        0.1:lambda a: a % 10 == 0,
        0.125:lambda a: a % 8 == 0,
        0.2:lambda a: a % 5 == 0,
        0.25:lambda a: a % 4 == 0,
        0.33:lambda a: a % 3 == 0,
        0.67:lambda a: a % 3 > 0,
        0.5:lambda a: a % 2 == 0,
        1.0:lambda a: True
    }

    fractions= [0.25] # , 0.1, 0.125, 0.2, 0.33] # , 0.33, 0.5, 0.67, 1.0] # [0.03, 0.04] # , 0.1, 0.33, 0.67, 1.0]
    use_zs=[True] # , False]
    models= [constants.MODEL_FULL] #  [constants.MODEL_CLS , constants.MODEL_FULL] #  , constants.MODEL_FULL] # , constants.MODEL_FULL constants.MODEL_CLS] # [constants.MODEL_FULL, constants.MODEL_CLS, constants.MODEL_VAE]

    p=Pool(2)
    params=[]
    max_epoch=2000
    epoch_checkpoint=0 # 300
    for cur_use_z in use_zs:
        for cur_fraction in fractions:
            for model in models:
                print("start {} {} use_z={}".format(cur_fraction, model, cur_use_z))
                # params.append([main, [model, cur_use_z, cur_fraction, max_epoch, epoch_checkpoint]])
                main(model=model, use_z=cur_use_z, fraction=cur_fraction, max_epoch=max_epoch, epoch_checkpoint=epoch_checkpoint)

    p.map(func_star, params)
