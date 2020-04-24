import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch.nn import functional as F
from torch import tensor

import constants_tcga as constants
import numpy as np
from multiprocessing import Pool
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


def plot(model, test_loader, device, suffix, path_to_save):
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
    X_pca = PCA(n_components=2).fit_transform(zs.detach().cpu().numpy())
    xs,ys=list(zip(*X_pca))
    plt.subplots(figsize=(20,20))
    labels=labels.cpu().numpy()
    plt.scatter(xs,ys, c=[a for a in labels], cmap='jet') # sns.color_palette("Paired", n_colors=len(constants.DATASETS_INCLUDED))[a]




    colormap = cm.jet
    label_unique = np.arange(len(constants.DATASETS_NAMES))
    colorlist_unique = [ml_colors.rgb2hex(colormap(a)) for a in
                    label_unique / float(max(labels))]
    patches = [Line2D([0], [0], marker='o', color='gray', label=constants.DATASETS_NAMES[a],
                  markerfacecolor=c) for a, c in zip(label_unique, colorlist_unique)]

    for a in label_unique:
        plt.scatter([np.mean([xs[i] for i, b in enumerate(labels) if a==b])],[np.mean([ys[i] for i, b in enumerate(labels) if a==b])], s=2000, c=colorlist_unique[a], cmap='jet', alpha=0.5)
        plt.annotate(constants.DATASETS_NAMES[a],
                    xy=(np.mean([xs[i] for i, b in enumerate(labels) if a==b]), np.mean([ys[i] for i, b in enumerate(labels) if a==b])), xytext=(-20, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=3, headlength=2))
    plt.legend(handles=patches) # plt.legend(handles=[a for a in constants.PATCHES])
    plt.savefig(os.path.join(path_to_save, "zs_scatter{}.png".format(suffix)))

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

    print('====> Epoch (VAE only): {} Average loss: {:.4f}'.format(epoch, train_vae_loss/n_batchs))

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


def main(model, use_z, fraction, max_epoch=300, epoch_checkpoint=0):

    fig,ax = plt.subplots()

    path_to_save_format=os.path.join(constants.CACHE_GLOBAL_DIR, constants.DATA_TYPE, "model_{}_{}_{}_{{}}".format(fraction,model,"z" if use_z else "mu"))
    path_to_plot=os.path.join(constants.OUTPUT_GLOBAL_DIR, "losses_{}_{}_{}_{}_{}.png".format(constants.DATA_TYPE, fraction,model,("z" if use_z else "mu"), max_epoch))
    train_losses=[]
    valid_losses=[]
    for cur_epoch in np.arange(epoch_checkpoint, max_epoch +1, 50):
        path_to_save = path_to_save_format.format(cur_epoch)

        train_losses=train_losses + [float(a.strip()) for a in open(os.path.join(path_to_save,"train_losses.txt"), 'r').readlines()]
        valid_losses=valid_losses + [float(a.strip()) for a in open(os.path.join(path_to_save,"val_losses.txt"), 'r').readlines()]

    ax_2=ax.twinx()
    ax.plot(np.arange(epoch_checkpoint, max_epoch + 1, 50), train_losses, label="train", color='blue')
    ax.plot([],[], label="valid", color='red')

    ax_2.plot(np.arange(epoch_checkpoint, max_epoch + 1, 50), valid_losses, color='red')
    ax.set_xlabel("# of epoch")
    # ax.set_yscale("log")
    ax.set_ylabel("loss")
    ax.legend()
    # ax_2.legend()
    plt.savefig(path_to_plot)


if __name__=="__main__":
    filter_func_dict={
        0.01:lambda a: True,
        0.05:lambda a: a % 20 == 0,
        0.1:lambda a: a % 10 == 0,
        0.33:lambda a: a % 3 == 0,
        0.67:lambda a: a % 3 > 0,
        1.0:lambda a: True
    }

    fractions=[1.0] # [0.1, 0.33, 0.67,1.0] # , 0.1, 0.33, 0.67, 1.0]
    use_zs=[True] # , False]
    models=  [constants.MODEL_VAE]

    p=Pool(2)
    params=[]
    max_epoch=50
    epoch_checkpoint=100 # 300
    for cur_use_z in use_zs:
        for cur_fraction in fractions:
            for model in models:
                print("start {} {} use_z={}".format(cur_fraction, model, cur_use_z))
                # params.append([main, [model, cur_use_z, cur_fraction, max_epoch, epoch_checkpoint]])
                main(model=model, use_z=cur_use_z, fraction=cur_fraction, max_epoch=max_epoch, epoch_checkpoint=epoch_checkpoint)

    # p.map(func_star, params)
