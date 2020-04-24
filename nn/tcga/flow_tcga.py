import matplotlib

matplotlib.use('Agg')

import torch
from torch.nn import functional as F


def loss_f_vae(recon_x, x, mu, logvar, epoch, factor):

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + KLD) * factor


def train_vae(epoch, encoder, decoder, factor, optimizer, train_loader, validation_loader, device, log_interval):

    n_batchs=0.0
    train_vae_loss=0
    for batch_idx, (data, labels) in enumerate(train_loader):
        n_batchs+=1

        data = data.to(device)
        optimizer.zero_grad()
        z, mu, logvar = encoder(data)
        recon_batch, z, mu, logvar = decoder((z, mu, logvar))

        loss_vae = loss_f_vae(recon_batch, data, mu, logvar, epoch, factor)
        loss_vae.backward()
        optimizer.step()
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
            valid_vae_loss+=loss_vae.item()

        print('====> Epoch (VAE only): {} Average validation loss: {:.4f}'.format(
              epoch, valid_vae_loss/n_batchs ))

    return [train_vae_loss/n_batchs], [valid_vae_loss/n_batchs]

def train_cls(epoch, encoder, classifier, factor, optimizer, train_loader, validation_loader, device, log_interval):

    n_batchs=0.0
    train_cls_loss=0
    for batch_idx, (data, labels) in enumerate(train_loader):
        n_batchs+=1
        if torch.sum(labels!=-1) == 0:
            continue

        data = data.to(device)
        optimizer.zero_grad()
        z, mu, logvar = encoder(data)
        labels_hat, z = classifier(z) # classifier(torch.cat((mu,logvar), dim=1))

        loss_cls = loss_cls(labels_hat, labels, labels != -1, epoch, factor)
        loss_cls.backward()
        optimizer.step()
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
            loss_cls = loss_cls(labels_hat, labels, labels != -1, epoch)
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

            loss_vae = loss_vae(recon_batch, data, mu, logvar)
            loss_cls = loss_cls(labels_hat, labels, labels != -1)


            train_vae_loss+=loss_vae.item()
            train_cls_loss+=loss_cls.item()


        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(test_loader.dataset),
            100. * batch_idx / len(test_loader),
            (loss_vae.item()+loss_cls.item())))

    print('====> Epoch: {} Average loss: {:.4f}, {:.4f}'.format(
          epoch, train_vae_loss/n_batchs , train_cls_loss/n_batchs ))
