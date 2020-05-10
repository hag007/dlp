import os

import matplotlib
matplotlib.use('Agg')

import torch

from nn.models import Encoder, Decoder, Classifier
import constants_tcga as constants
from datasets import datasets
import torch.optim as optim
import numpy as np
import copy
from multiprocessing import Pool

from nn.tcga.flow_tcga import train_vae
from plots.scatter_plot_test import plot

filter_func_dict={
        0.01:lambda a: True,
        0.05:lambda a: a % 20 == 0,
        0.1:lambda a: a % 10 == 0,
        0.33:lambda a: a % 3 == 0,
        0.67:lambda a: a % 3 > 0,
        1.0:lambda a: True
    }

def main(model, use_z, fraction, max_epoch=300, epoch_checkpoint=0):

    filter_func=filter_func_dict[fraction]

    n_latent_layer=2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset_names=constants.ALL_DATASET_NAMES
    dataset_mask= datasets.DatasetMask(dataset_names, constants.DATA_TYPE, filter_func)
    dataloader_ctor_mask= datasets.DataLoader(dataset_mask, 0.2, 0, 2)
    trainloader = dataloader_ctor_mask.train_loader()
    validationloader = dataloader_ctor_mask.valid_loader()

    dataset= datasets.Dataset(dataset_names, constants.DATA_TYPE)
    dataloader_ctor= datasets.DataLoader(dataset, 0.2, 0.2)
    testloader = dataloader_ctor.train_loader()

    encoder=Encoder(n_latent_layer=n_latent_layer)
    decoder=Decoder(n_latent_layer=n_latent_layer)

    path_format_to_save=os.path.join(constants.CACHE_GLOBAL_DIR, constants.DATA_TYPE, "model_{}_{}_{}_{{}}".format(fraction,model,"z" if use_z else "mu"))
    PATH_ENCODER= os.path.join(path_format_to_save,"ENC_mdl")
    PATH_DECODER= os.path.join(path_format_to_save,"DEC_mdl")

    load_model=epoch_checkpoint >0
    if load_model and os.path.exists(PATH_ENCODER.format(epoch_checkpoint)):
        encoder.load_state_dict(torch.load(PATH_ENCODER.format(epoch_checkpoint)))
        encoder.eval()
        decoder.load_state_dict(torch.load(PATH_DECODER.format(epoch_checkpoint)))
        decoder.eval()
    else:
        epoch_checkpoint=0


    lr = 1e-4
    parameters=list(encoder.parameters())+list(decoder.parameters())
    optimizer_vae = optim.Adam(parameters, lr=lr)
    log_interval=100
    min_encoder=None
    min_decoder=None
    min_epoch=-1
    min_val_loss=10e10
    train_losses=[]
    val_losses=[]
    for cur_epoch in np.arange(epoch_checkpoint, max_epoch + 1):

        train_loss, validation_loss, = train_vae(cur_epoch, encoder, decoder, 1, optimizer_vae, trainloader, validationloader, device, log_interval)
        train_losses.append(['{:.2f}'.format(a) for a in train_loss])
        val_losses.append(['{:.2f}'.format(a) for a in validation_loss])

        if min_val_loss>sum([validation_loss[-1]]):
            min_encoder = copy.deepcopy(encoder)
            min_decoder = copy.deepcopy(decoder)

            min_epoch=cur_epoch
            min_val_loss=sum([validation_loss[-1]])

        print("min_val_loss: {} (epoch n={})".format(min_epoch, min_val_loss))

        if (cur_epoch) % 50 == 0:
            try:
                os.makedirs(path_format_to_save.format(cur_epoch))
            except:
                pass
            if min_encoder is not None:
                torch.save(min_encoder.state_dict(), PATH_ENCODER.format(cur_epoch)+"_min")
                torch.save(min_decoder.state_dict(), PATH_DECODER.format(cur_epoch)+"_min")
                open(os.path.join(path_format_to_save.format(cur_epoch), "min_epoch.txt"), "w").write("{}_{}".format(min_val_loss,min_epoch))
                plot(min_encoder, testloader, device, "_min", dataset_names, path_format_to_save.format(cur_epoch))

            torch.save(encoder.state_dict(), PATH_ENCODER.format(cur_epoch))
            torch.save(decoder.state_dict(), PATH_DECODER.format(cur_epoch))

            plot(encoder, testloader, device, "", dataset_names, path_format_to_save.format(cur_epoch))

            open(os.path.join(path_format_to_save.format(cur_epoch), "train_losses.txt"), "w").write("\n".join(train_losses[0]))
            open(os.path.join(path_format_to_save.format(cur_epoch), "val_losses.txt"), "w").write("\n".join(val_losses[0]))
            train_losses=[]
            val_losses=[]

if __name__=="__main__":


    fractions=[1.0]
    use_zs=[True]
    models=  [constants.MODEL_VAE]

    p=Pool(2)
    params=[]
    max_epoch=10000+1
    epoch_checkpoint=0
    for cur_use_z in use_zs:
        for cur_fraction in fractions:
            for model in models:
                print("start {} {} use_z={}".format(cur_fraction, model, cur_use_z))
                main(model=model, use_z=cur_use_z, fraction=cur_fraction, max_epoch=max_epoch, epoch_checkpoint=epoch_checkpoint)