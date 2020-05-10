import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch.nn import functional as F
from torch import tensor

from nn.models import Encoder, Decoder, Classifier
import constants_tcga as constants
from datasets import datasets
import torch.optim as optim
import numpy as np
import seaborn as sns
import copy
from multiprocessing import Pool
from sklearn.decomposition import PCA

def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return a_b[0](*a_b[1])

def loss_f_vae(recon_x, x, mu, logvar, epoch, factor):

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + KLD)*factor

def loss_f_cls(labels_hat, labels, mask, epoch, factor):
    labels_hat_masked=labels_hat[mask]
    labels_masked=labels[mask]
    CE=torch.nn.CrossEntropyLoss( reduction='sum')(labels_hat_masked, labels_masked) *mask.shape[0]/float(torch.sum(mask))# *((2000.0/labels_hat_masked.shape[1])*float(mask.shape[0])/torch.sum(mask))

    return CE * factor# *torch.sum(mask)/float(mask.shape[0])


def train_full(epoch, encoder, decoder, classifier, factor_vae, factor_cls, optimizer_vae, optimizer_cls, train_loader, validation_loader, device, log_interval):

    n_batchs=0.0
    train_loss_cls_agg=0
    train_loss_vae_agg=0
    for batch_idx, (data, labels) in enumerate(train_loader):
        n_batchs+=1

        data = data.to(device)
        optimizer_vae.zero_grad()
        optimizer_cls.zero_grad()
        z, mu, logvar, _ = encoder(data)
        recon_batch, z, mu, logvar = decoder((z, mu, logvar))
        loss_train_vae = loss_f_vae(recon_batch, data, mu, logvar, epoch, factor_vae)
        train_loss_vae_agg+= loss_train_vae.item()

        if torch.sum(labels!=-1) != 0:
            labels_hat, _, _, reduction_layer = classifier(z) #
            # labels_hat, _, _, reduction_layer = classifier(torch.cat((mu,logvar), dim=1))
            loss_valid_cls = loss_f_cls(labels_hat, labels, labels != -1, epoch, factor_cls)
            train_loss_cls_agg+=loss_valid_cls.item()

        loss_train_vae.backward(retain_graph=True)
        optimizer_vae.step()
        if torch.sum(labels!=-1) != 0:
            loss_valid_cls.backward()
            optimizer_cls.step()

        if (batch_idx+1) % log_interval == 0:
            if torch.sum(labels!=-1) != 0:
                print('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    (loss_valid_cls.item()+loss_train_vae.item())))

            else:
                print('Train Epoch {} (VAE only): [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    (loss_train_vae.item())))

    print('====> Epoch: {} Average train loss: {:.4f}, {:.4f}'.format(
          epoch, train_loss_vae_agg/n_batchs , train_loss_cls_agg/n_batchs ))

    n_batchs=0.0
    loss_valid_cls_agg=0
    loss_valid_vae_agg=0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(validation_loader):
            n_batchs+=1

            data = data.to(device)
            z, mu, logvar, _ = encoder(data)
            recon_batch, z, mu, logvar = decoder((z, mu, logvar))
            loss_valid_vae = loss_f_vae(recon_batch, data, mu, logvar, epoch, factor_vae)
            loss_valid_vae_agg+=loss_valid_vae.item()

            if torch.sum(labels!=-1) != 0:
                labels_hat, _, _, reduction_layer = classifier(z)
                # labels_hat, z = classifier(torch.cat((mu,logvar), dim=1))
                loss_valid_cls = loss_f_cls(labels_hat, labels, labels != -1, epoch, factor_cls)
                loss_valid_cls_agg+=loss_valid_cls.item()

        print('====> Epoch: {} Average validation loss: {:.4f}, {:.4f}'.format(
              epoch, loss_valid_vae/n_batchs , loss_valid_cls_agg/n_batchs ))

    return [train_loss_vae_agg/n_batchs,train_loss_cls_agg/n_batchs],  [loss_valid_vae_agg/n_batchs,loss_valid_cls_agg/n_batchs]


def train_vae(epoch, encoder, decoder, classifier, factor_vae, factor_cls, optimizer_vae, optimizer_cls, train_loader, validation_loader, device, log_interval):

    n_batchs=0.0
    loss_train_agg=0
    for batch_idx, (data, labels) in enumerate(train_loader):
        n_batchs+=1

        data = data.to(device)
        optimizer_vae.zero_grad()
        z, mu, logvar, _ = encoder(data)
        recon_batch, z, mu, logvar = decoder((z, mu, logvar))

        loss_train = loss_f_vae(recon_batch, data, mu, logvar, epoch, factor_vae)
        loss_train_agg+= loss_train.item()
        loss_train.backward()
        optimizer_vae.step()

        if (batch_idx+1) % log_interval == 0:
            print('Train Epoch (VAE only): {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                (loss_train.item())))

    print('====> Epoch {} (VAE only): Average loss: {:.4f}'.format(epoch, loss_train_agg/n_batchs))

    n_batchs=0.0
    loss_valid_agg=0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(validation_loader):
            n_batchs+=1

            data = data.to(device)
            z, mu, logvar, _ = encoder(data)
            recon_batch, z, mu, logvar = decoder((z, mu, logvar))
            loss_valid = loss_f_vae(recon_batch, data, mu, logvar, epoch, factor_vae)
            loss_valid_agg+=loss_valid.item()

        print('====> Epoch (VAE only): {} Average validation loss: {:.4f}'.format(
              epoch, loss_valid/n_batchs ))

    return [loss_train_agg/n_batchs], [loss_valid_agg/n_batchs]

def train_cls(epoch, encoder, decoder, classifier, factor_vae, factor_cls, optimizer_vae, optimizer_cls, train_loader, validation_loader, device, log_interval):

    n_batchs=0.0
    loss_train_agg=0
    for batch_idx, (data, labels) in enumerate(train_loader):
        n_batchs+=1
        if torch.sum(labels!=-1) == 0:
            continue

        data = data.to(device)
        optimizer_cls.zero_grad()
        z, mu, logvar, _ = encoder(data)
        # labels_hat, z = classifier(z)
        # labels_hat, _, _, reduction_layer = classifier(torch.cat((mu,logvar), dim=1))
        labels_hat, _, _, reduction_layer = classifier(z)
        loss_train = loss_f_cls(labels_hat, labels, labels != -1, epoch, factor_cls)
        loss_train.backward()
        optimizer_cls.step()
        loss_train_agg+=loss_train.item()

        if (batch_idx+1) % log_interval == 0:
            print('Train Epoch (CLS only): {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                (loss_train.item())))

    print('====> Epoch (CLS only): {} Average loss: {:.4f}'.format(
          epoch , loss_train_agg/n_batchs ))

    n_batchs=0.0
    loss_valid_cls_agg=0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(validation_loader):
            n_batchs+=1
            if torch.sum(labels!=-1) == 0:
                continue

            data = data.to(device)
            z, mu, logvar, _ = encoder(data)

            labels_hat, _, _, reduction_layer =  classifier(z)
            # labels_hat, _, _, reduction_layer = classifier(torch.cat((mu,logvar), dim=1))
            loss_valid = loss_f_cls(labels_hat, labels, labels != -1, epoch, factor_cls)
            loss_valid_cls_agg+=loss_valid.item()

        print('====> Epoch: {} Average validation loss:  {:.4f}'.format(
              epoch , loss_valid_cls_agg/n_batchs ))

    return [loss_train_agg/n_batchs], [loss_valid_cls_agg/n_batchs]

#
# def test(epoch, encoder,decoder,classifier, test_loader, device):
#
#     with torch.no_grad():
#         n_batchs=0.0
#         loss_train_cls=0
#         loss_train_vae=0
#         for batch_idx, (data, labels) in enumerate(test_loader):
#             n_batchs+=1
#             data = data.to(device)
#             z, mu, logvar = encoder(data)
#             recon_batch, z, mu, logvar = decoder((z, mu, logvar))
#             labels_hat, z = classifier(z)
#
#             loss_vae = loss_f_vae(recon_batch, data, mu, logvar)
#             loss_cls = loss_f_cls(labels_hat, labels, labels != -1)
#
#             loss_train_vae+=loss_vae.item()
#             loss_train_cls+=loss_cls.item()
#
#
#     print('====> Epoch: {} Average loss: {:.4f}, {:.4f}'.format(
#           epoch, loss_train_vae/n_batchs , loss_train_cls/n_batchs ))
#
