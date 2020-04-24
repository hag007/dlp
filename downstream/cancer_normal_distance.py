import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch.nn import functional as F
from torch import tensor

from nn.models import Encoder, Decoder, Classifier
import constants_tcga as constants
from datasets import cmap_datasets
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.cm as cm
import matplotlib.colors as ml_colors
from matplotlib.lines import Line2D

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


def plot(model, test_loader, device, suffix, path_to_save, dataset_names, colormap, bg_color):
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

    np.save(os.path.join(path_to_save, "latent_features{}.npy".format(suffix)), np.hstack([zs, labels.reshape(-1,1), [[constants.DATASETS_NAMES[a]] for a in labels]]))
    X_pca = PCA(n_components=2).fit_transform(zs)

    xs,ys=list(zip(*X_pca))

    plt.scatter(xs,ys, c=[a for a in labels], cmap=colormap) # sns.color_palette("Paired", n_colors=len(constants.DATASETS_INCLUDED))[a]


    label_unique = np.arange(len(dataset_names))
    colorlist_unique = [ml_colors.rgb2hex(colormap(a)) for a in
                    label_unique / float(max(labels))]
    patches = [Line2D([0], [0], marker='o', color='gray', label=dataset_names[a],
                  markerfacecolor=c) for a, c in zip(label_unique, colorlist_unique)]

    for a in label_unique:
        plt.scatter([np.median([xs[i] for i, b in enumerate(labels) if a==b])],[np.median([ys[i] for i, b in enumerate(labels) if a==b])], s=2000, c=colorlist_unique[a], cmap=colormap, alpha=0.5)
        plt.annotate(dataset_names[a],
                    xy=(np.median([xs[i] for i, b in enumerate(labels) if a==b]), np.median([ys[i] for i, b in enumerate(labels) if a==b])), xytext=(-20, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc=bg_color, alpha=0.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=3, headlength=2))
    return patches

def plot_median_diff(model, test_loader, device, suffix, path_to_save, dataset_names, colormap, bg_color):
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

    np.save(os.path.join(path_to_save, "latent_features{}.npy".format(suffix)), np.hstack([zs, labels.reshape(-1,1), [[constants.DATASETS_NAMES[a]] for a in labels]]))
    X_pca = PCA(n_components=2).fit_transform(zs)

    xs,ys=list(zip(*X_pca))

    label_unique = np.arange(len(dataset_names))

    labels_unique_half=np.arange(int(np.ceil(len(dataset_names)/2.0)))
    colorlist_unique = [ml_colors.rgb2hex(colormap(a)) for a in
                    labels_unique_half / float(max(labels)/2)]
    patches = [Line2D([0], [0], marker='o', color='gray', label=dataset_names[a],
                  markerfacecolor=c) for a, c in zip(labels_unique_half, colorlist_unique)]

    cancer_dict_tumor={}
    cancer_dict_normal={}
    for a in label_unique:
        # plt.scatter([np.median([xs[i] for i, b in enumerate(labels) if a==b])],[np.median([ys[i] for i, b in enumerate(labels) if a==b])], s=2000, c=colorlist_unique[a], cmap=colormap, alpha=0.5)
        # plt.annotate(dataset_names[a],
        #             xy=(np.median([xs[i] for i, b in enumerate(labels) if a==b]), np.median([ys[i] for i, b in enumerate(labels) if a==b])), xytext=(-20, 20), textcoords='offset points',
        #             bbox=dict(boxstyle='round,pad=0.5', fc=bg_color, alpha=0.5),
        #             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=3, headlength=2))
        if "tumor" in dataset_names[a]:
            cancer_dict_tumor[dataset_names[a].split("_")[0]]=np.array([np.median([xs[i] for i, b in enumerate(labels) if a==b]), np.median([ys[i] for i, b in enumerate(labels) if a==b])])
        else:
            cancer_dict_normal[dataset_names[a].split("_")[0]]=np.array([np.median([xs[i] for i, b in enumerate(labels) if a==b]), np.median([ys[i] for i, b in enumerate(labels) if a==b])])

    for i, k in enumerate(cancer_dict_tumor):
        diff=cancer_dict_tumor[k]-cancer_dict_normal[k]
        plt.scatter([diff[0]],[diff[1]], s=2000, c=colorlist_unique[i], cmap=colormap, alpha=0.5)
        plt.annotate(k+"_diff",
                    xy=(diff[0], diff[1]), xytext=(-20, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc=bg_color, alpha=0.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=3, headlength=2))



    return patches

def main(model,use_z, fraction, epoch_checkpoint=300, suffix=""):

    filter_func=filter_func_dict[fraction]

    n_latent_layer=2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    genes=None
    genes_name=None
    # genes=np.load("/media/hag007/Data/dlproj/cache_global/datasets/vemurafenib_resveratrol_olaparib/genes.npy", allow_pickle=True)
    # genes_name="vemurafenib_resveratrol_olaparib"

    dataset_names=[a for i, a in enumerate(constants.DATASETS_NAMES) if constants.DATASETS_NAMES[i]]
    dataset= cmap_datasets.CMAPDataset(genes, genes_name, dataset_names, "tcga")
    dataloader_ctor= cmap_datasets.CMAPDataLoader(dataset, 0.0, 0.0)
    testloader = dataloader_ctor.train_loader()

    dataset_names=[a for i, a in enumerate(constants.DATASETS_NAMES) if constants.DATASETS_INCLUDED[i]]
    dataset= cmap_datasets.CMAPDataset(genes, genes_name, dataset_names, "tcga")
    dataloader_ctor= cmap_datasets.CMAPDataLoader(dataset, 0.0, 0.0)
    trainloader = dataloader_ctor.train_loader()

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
        encoder.eval()
        decoder.load_state_dict(torch.load(PATH_DECODER.format(epoch_checkpoint)+suffix))
        decoder.eval()
        if model != constants.MODEL_VAE:
            classifier.load_state_dict(torch.load(PATH_CLASSIFIER.format(epoch_checkpoint)+suffix))
            classifier.eval()

    with torch.no_grad():
        path_to_save=path_format_to_save.format(epoch_checkpoint)
        plt.subplots(figsize=(20,20))
        colormap = cm.jet
        patches_tcga=plot(encoder, trainloader, device, suffix + "_tcga", path_to_save, constants.DATASETS_NAMES, colormap, 'yellow')
        # plt.legend(handles=patches_tcga)
        # plt.savefig(os.path.join(path_to_save, "zs_scatter{}.png".format(suffix + "_tcga_diff")))
        plt.clf()

        plt.subplots(figsize=(20,20))
        colormap = cm.jet
        patches_tcga=plot_median_diff(encoder, trainloader, device, suffix + "_tcga", path_to_save, constants.DATASETS_NAMES, colormap, 'yellow')
        plt.legend(handles=patches_tcga)
        plt.savefig(os.path.join(path_to_save, "zs_scatter{}.png".format(suffix + "_tcga_diff")))


if __name__=="__main__":
    filter_func_dict={
        0.01:lambda a: a % 100 == 0,
        0.05:lambda a: a % 20 == 0,
        0.1:lambda a: a % 10 == 0,
        0.33:lambda a: a % 3 == 0,
        0.67:lambda a: a % 3 > 0,
        1.0:lambda a: True
    }

    fractions=[1.0] # , 0.33, 0.67, 1.0]
    use_zs=[True] # , False]
    models=[constants.MODEL_VAE]

    epoch_checkpoints=[5150] # [50, 100, 150,200,250,300]
    suffix=""
    for cur_use_z in use_zs:
        for cur_fraction in fractions:
            for model in models:
                for cur_epoch_checkpoint in epoch_checkpoints:
                    print("start {} {} use_z={}".format(cur_fraction, model, cur_use_z))
                    # params.append([main, [model, cur_use_z, cur_fraction, epoch_checkpoint, "_min"]])
                    main(model=model, use_z=cur_use_z, fraction=cur_fraction, epoch_checkpoint=cur_epoch_checkpoint, suffix=suffix)

    # p.map(func_star, params)
