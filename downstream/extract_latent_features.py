import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch.nn import functional as F
from torch import tensor

from nn.models import Encoder, Decoder, Classifier
import constants
from datasets import cmap_datasets
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return a_b[0](*a_b[1])

def loss_function(recon_x, x, mu, logvar, epoch):

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + KLD)

def loss_function_cls(labels_hat, labels, mask, epoch):
    labels_hat_masked=labels_hat[mask]
    labels_masked=labels[mask]
    CE=torch.nn.CrossEntropyLoss( reduction='sum')(labels_hat_masked, labels_masked) *((2000.0/labels_hat_masked.shape[1])*float(mask.shape[0])/torch.sum(mask))

    return CE/10


def extract_latent_dimension(model, test_loader, device, suffix, path_to_save):
    zs=tensor([])
    mus=tensor([])
    logvars=tensor([])
    labels=tensor([]).long()

    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(device)
        z, mu, logvar = model(data)
        zs=torch.cat((zs, z), 0)
        mus=torch.cat((mus, mu), 0)
        logvars=torch.cat((logvars, logvar), 0)
        labels=torch.cat((labels, label), 0)

    # xs,ys=list(zip(*zs.cpu().numpy()))
    zs=zs.cpu().numpy()
    labels=labels.cpu().numpy()
    zs[np.isneginf(zs)]=-1000000000
    zs[np.isposinf(zs)]=1000000000
    np.save(os.path.join(path_to_save, "latent_features{}.npy".format(suffix)), np.hstack([zs, labels.reshape(-1,1), [[constants.DATASETS_NAMES[a]] for a in labels]]))
    X_pca = PCA(n_components=2).fit_transform(zs)

    xs,ys=list(zip(*X_pca))
    plt.subplots(figsize=(20,20))
    plt.scatter(xs,ys, c=[sns.color_palette("Paired", n_colors=len(constants.DATASETS_INCLUDED))[a] for a in labels])
    plt.legend(handles=[a for a in constants.PATCHES])
    plt.savefig(os.path.join(path_to_save, "latent_scatter{}.png".format(suffix)))

def train_full(epoch, encoder, decoder, classifier, vae_optimizer, cls_optimizer, train_loader, validation_loader, device, log_interval):

    n_batchs=0.0
    train_cls_loss=0
    train_vae_loss=0
    for batch_idx, (data, labels) in enumerate(train_loader):
        n_batchs+=1

        data = data.to(device)
        vae_optimizer.zero_grad()
        cls_optimizer.zero_grad()
        z, mu, logvar = encoder(data)
        recon_batch, z, mu, logvar = decoder((z, mu, logvar))
        # if torch.sum(labels!=-1) != 0:

        loss_vae = loss_function(recon_batch, data, mu, logvar, epoch)
        loss_vae.backward(retain_graph=True)

        if torch.sum(labels!=-1) != 0:
            labels_hat, z = classifier(z) # classifier(torch.cat((mu,logvar), dim=1))
            loss_cls = loss_function_cls(labels_hat, labels, labels!=-1, epoch)
            loss_cls.backward()

        vae_optimizer.step()
        cls_optimizer.step()


        train_vae_loss+= loss_vae.item()
        if torch.sum(labels!=-1) != 0:
            train_cls_loss+=loss_cls.item()


        if (batch_idx+1) % log_interval == 0:
            if torch.sum(labels!=-1) != 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    (loss_cls.item()+loss_vae.item())))

            else:
                print('Train Epoch (VAE only): {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    (loss_vae.item())))

    print('====> Epoch: {} Average train loss: {:.4f}, {:.4f}'.format(
          epoch, train_vae_loss/n_batchs , train_cls_loss/n_batchs ))

    n_batchs=0.0
    valid_cls_loss=0
    valid_vae_loss=0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(validation_loader):
            n_batchs+=1

            data = data.to(device)
            z, mu, logvar = encoder(data)
            recon_batch, z, mu, logvar = decoder((z, mu, logvar))
            loss_vae = loss_function(recon_batch, data, mu, logvar, epoch)
            valid_vae_loss+=loss_vae.item()

            if torch.sum(labels!=-1) != 0:
                labels_hat, z = classifier(z) # classifier(torch.cat((mu,logvar), dim=1))
                loss_cls = loss_function_cls(labels_hat, labels, labels!=-1, epoch)
                valid_cls_loss+=loss_cls.item()



        print('====> Epoch: {} Average validation loss: {:.4f}, {:.4f}'.format(
              epoch, valid_vae_loss/n_batchs , valid_cls_loss/n_batchs ))

    return [train_vae_loss/n_batchs,train_cls_loss/n_batchs],  [valid_vae_loss/n_batchs,valid_cls_loss/n_batchs]


def train_vae(epoch, encoder,decoder,classifier, vae_optimizer, cls_optimizer, train_loader, validation_loader, device, log_interval):

    n_batchs=0.0
    train_vae_loss=0
    for batch_idx, (data, labels) in enumerate(train_loader):
        n_batchs+=1

        data = data.to(device)
        vae_optimizer.zero_grad()
        z, mu, logvar = encoder(data)
        recon_batch, z, mu, logvar = decoder((z, mu, logvar))

        loss_vae = loss_function(recon_batch, data, mu, logvar, epoch)
        loss_vae.backward()
        vae_optimizer.step()
        train_vae_loss+= loss_vae.item()

        if (batch_idx+1) % log_interval == 0:
            print('Train Epoch (VAE only): {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                (loss_vae.item())))

    print('====> Epoch (VAE only): {} Average loss: {:.4f}'.format(
          (epoch, train_vae_loss/n_batchs)))

    n_batchs=0.0
    valid_vae_loss=0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(validation_loader):
            n_batchs+=1

            data = data.to(device)
            z, mu, logvar = encoder(data)
            recon_batch, z, mu, logvar = decoder((z, mu, logvar))
            loss_vae = loss_function(recon_batch, data, mu, logvar, epoch)
            valid_vae_loss+=loss_vae.item()

        print('====> Epoch (VAE only): {} Average validation loss: {:.4f}'.format(
              epoch, valid_vae_loss/n_batchs ))

    return [train_vae_loss/n_batchs], [valid_vae_loss/n_batchs]

def train_cls(epoch, encoder,decoder,classifier, vae_optimizer, cls_optimizer, train_loader, validation_loader, device, log_interval):

    n_batchs=0.0
    train_cls_loss=0
    for batch_idx, (data, labels) in enumerate(train_loader):
        n_batchs+=1
        if torch.sum(labels!=-1) == 0:
            continue

        data = data.to(device)
        cls_optimizer.zero_grad()
        z, mu, logvar = encoder(data)
        labels_hat, z = classifier(z) # classifier(torch.cat((mu,logvar), dim=1))

        loss_cls = loss_function_cls(labels_hat, labels, labels!=-1, epoch)
        loss_cls.backward()
        cls_optimizer.step()
        train_cls_loss+=loss_cls.item()

        if (batch_idx+1) % log_interval == 0:
            print('Train Epoch (CLS only): {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                (loss_cls.item())))

    print('====> Epoch (CLS only): {} Average loss: {:.4f}'.format(
          epoch , train_cls_loss/n_batchs ))

    n_batchs=0.0
    valid_cls_loss=0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(validation_loader):
            n_batchs+=1
            if torch.sum(labels!=-1) == 0:
                continue

            data = data.to(device)
            z, mu, logvar = encoder(data)

            labels_hat, z = classifier(z) # classifier(torch.cat((mu,logvar), dim=1))
            loss_cls = loss_function_cls(labels_hat, labels, labels!=-1, epoch)
            valid_cls_loss+=loss_cls.item()

        print('====> Epoch: {} Average validation loss:  {:.4f}'.format(
              epoch , valid_cls_loss/n_batchs ))

    return [train_cls_loss/n_batchs], [valid_cls_loss/n_batchs]


def test(epoch, encoder,decoder,classifier, test_loader, device):

    with torch.no_grad():
        n_batchs=0.0
        train_cls_loss=0
        train_vae_loss=0
        for batch_idx, (data, labels) in enumerate(test_loader):
            n_batchs+=1
            data = data.to(device)
            z, mu, logvar = encoder(data)
            recon_batch, z, mu, logvar = decoder((z, mu, logvar))
            labels_hat, z = classifier(z)

            loss_vae = loss_function(recon_batch, data, mu, logvar)
            loss_cls = loss_function_cls(labels_hat, labels, labels!=-1)


            train_vae_loss+=loss_vae.item()
            train_cls_loss+=loss_cls.item()


        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(test_loader.dataset),
            100. * batch_idx / len(test_loader),
            (loss_vae.item()+loss_cls.item())))

    print('====> Epoch: {} Average loss: {:.4f}, {:.4f}'.format(
          epoch, train_vae_loss/n_batchs , train_cls_loss/n_batchs ))


def main(model,use_z, fraction, epoch_checkpoint=300, suffix=""):

    filter_func=filter_func_dict[fraction]

    n_latent_layer=2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    genes=None
    genes_name=None
    # genes=np.load("/media/hag007/Data/dlproj/cache_global/datasets/vemurafenib_resveratrol_olaparib/genes.npy", allow_pickle=True)
    # genes_name="vemurafenib_resveratrol_olaparib"


    dataset= cmap_datasets.CMAPDataset(genes, genes_name)
    dataloader_ctor= cmap_datasets.CMAPDataLoader(dataset, 0.2, 0.2)
    testloader = dataloader_ctor.test_loader()

    dataloader_ctor_mask= cmap_datasets.CMAPDataLoader(dataset, 0.2, 0, 2)
    trainloader = dataloader_ctor_mask.train_loader()
    validationloader = dataloader_ctor_mask.valid_loader()

    encoder=Encoder(n_latent_layer=n_latent_layer)
    decoder=Decoder(n_latent_layer=n_latent_layer)
    classifier=Classifier(n_latent_layer=n_latent_layer, n_classes=(len(constants.DATASETS_FILES) if genes_name is None else genes_name.count("_")+1))

    path_format_to_save=os.path.join(constants.CACHE_GLOBAL_DIR, constants.DATA_TYPE, "model_{}_{}_{}_{{}}".format(fraction,model,"z" if use_z else "mu"))
    PATH_ENCODER= os.path.join(path_format_to_save,"ENC_mdl")
    PATH_DECODER= os.path.join(path_format_to_save,"DEC_mdl")
    PATH_CLASSIFIER= os.path.join(path_format_to_save,"CLS_mdl")

    load_model=True
    if load_model and os.path.exists(PATH_ENCODER.format(epoch_checkpoint)+suffix):
        encoder.load_state_dict(torch.load(PATH_ENCODER.format(epoch_checkpoint)+suffix))
        decoder.load_state_dict(torch.load(PATH_DECODER.format(epoch_checkpoint)+suffix))
        classifier.load_state_dict(torch.load(PATH_CLASSIFIER.format(epoch_checkpoint)+suffix))
        encoder.eval()
        decoder.eval()
        classifier.eval()

    with torch.no_grad():
        extract_latent_dimension(encoder, trainloader, device, suffix + "_train", path_format_to_save.format(epoch_checkpoint))
        extract_latent_dimension(encoder, validationloader, device, suffix + "_validation", path_format_to_save.format(epoch_checkpoint))
        extract_latent_dimension(encoder, testloader, device, suffix + "_test", path_format_to_save.format(epoch_checkpoint))


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
    suffix= ""
    for cur_use_z in use_zs:
        for cur_fraction in fractions:
            for model in models:
                for cur_epoch_checkpoint in epoch_checkpoints:
                    print("start {} {} use_z={}".format(cur_fraction, model, cur_use_z))
                    # params.append([main, [model, cur_use_z, cur_fraction, epoch_checkpoint, "_min"]])
                    main(model=model, use_z=cur_use_z, fraction=cur_fraction, epoch_checkpoint=cur_epoch_checkpoint, suffix=suffix)

    # p.map(func_star, params)
