import matplotlib
matplotlib.use('Agg')

import torch
from torch.nn import functional as F

def loss_f_vae(recon_x, x, mu, logvar, epoch, factor):

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + KLD) * factor


def train_vae(epoch, encoder, decoder, factor, optimizer, train_loader, valid_loader, device, log_interval):

    n_batchs=0.0
    loss_train_agg=0
    for batch_idx, (data, labels) in enumerate(train_loader):
        n_batchs+=1

        data = data.to(device)
        optimizer.zero_grad()
        z, mu, logvar, _  = encoder(data)
        recon_batch, z, mu, logvar = decoder((z, mu, logvar))

        loss_train = loss_f_vae(recon_batch, data, mu, logvar, epoch, factor)
        loss_train.backward()
        optimizer.step()
        loss_train_agg+= loss_train.item()


        if (batch_idx+1) % log_interval == 0:
            print('Train Epoch {} (VAE only): [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                (loss_train.item())))

    print('====> Epoch {} (VAE only): Average loss: {:.4f}'.format(epoch, loss_train_agg/n_batchs))

    n_batchs=0.0
    loss_valid_agg=0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(valid_loader):
            n_batchs+=1

            data = data.to(device)
            z, mu, logvar, _ = encoder(data)
            recon_batch, z, mu, logvar = decoder((z, mu, logvar))
            loss_valid = loss_f_vae(recon_batch, data, mu, logvar, epoch, factor)
            optimizer.step()
            loss_valid_agg+= loss_valid.item()

        print('====> Epoch {} (VAE only): Average validation loss: {:.4f}'.format(
              epoch, loss_valid_agg/n_batchs ))

    return [loss_train_agg/n_batchs], [loss_valid_agg/n_batchs]

# def test(epoch, encoder,decoder, test_loader, device):
#
#     with torch.no_grad():
#         n_batchs=0.0
#         loss_test_agg=0
#         for batch_idx, (data, labels) in enumerate(test_loader):
#             n_batchs+=1
#             data = data.to(device)
#             z, mu, logvar = encoder(data)
#             recon_batch, z, mu, logvar = decoder((z, mu, logvar))
#             loss_test = loss_test(recon_batch, data, mu, logvar)
#
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(test_loader.dataset),
#                 100. * batch_idx / len(test_loader),
#                 loss_test.item()))
#
#         print('====> Epoch: {} Average loss: {:.4f}, {:.4f}'.format(
#               epoch, loss_test_agg/n_batchs , loss_test_agg/n_batchs ))
