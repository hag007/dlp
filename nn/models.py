import argparse
import numpy as np
import os
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

batch_size_train = 100
batch_size_val = 10
num_workers = 25
n_input_layer= 2000 # 1000


class Encoder(nn.Module):
    def __init__(self, factor=0.5, n_input_layer=n_input_layer,
                 n_latent_layer=100, n_reduction_layers=2):
        super(Encoder, self).__init__()
        self.n_reduction_layers = n_reduction_layers
        self.n_latent_vector = n_latent_layer

        for cur in np.arange(1, n_reduction_layers + 1):
            setattr(self, "fc_enc" + str(cur),
                    nn.Linear(int(n_input_layer * factor ** (cur - 1)), int(n_input_layer * factor ** cur)))
            setattr(self, "fc_bn_enc" + str(cur), nn.BatchNorm1d(int(n_input_layer * factor ** cur)))

        self.fc_enc_l_mu = nn.Linear(int(n_input_layer * factor ** n_reduction_layers), n_latent_layer)
        self.fc_bn_enc_l_mu = nn.BatchNorm1d(n_latent_layer)
        self.fc_enc_l_var = nn.Linear(int(n_input_layer * factor ** n_reduction_layers), n_latent_layer)
        self.fc_bn_enc_l_var = nn.BatchNorm1d(n_latent_layer)

    def encode(self, x):
        h = x
        for cur in np.arange(1, self.n_reduction_layers + 1):
            h = getattr(self, "fc_bn_enc" + str(cur))(F.relu(getattr(self, "fc_enc" + str(cur))(h)))

        l_mu = getattr(self, "fc_bn_enc_l_mu")(F.relu(getattr(self, "fc_enc_l_mu")(h)))
        l_var = getattr(self, "fc_bn_enc_l_var")(F.relu(getattr(self, "fc_enc_l_var")(h)))

        return l_mu, l_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, z


class Decoder(nn.Module):

    def __init__(self, factor=0.5, n_input_layer=n_input_layer,
                 n_latent_layer=100, n_reduction_layers=2):
        super(Decoder, self).__init__()
        self.n_reduction_layers = n_reduction_layers
        self.n_latent_vector = n_latent_layer

        self.fc_dec_l = nn.Linear(n_latent_layer, int(n_input_layer * factor ** n_reduction_layers))
        self.fc_bn_dec_l = nn.BatchNorm1d(int(n_input_layer * factor ** n_reduction_layers))

        for cur in np.arange(n_reduction_layers, 1, -1):
            setattr(self, "fc_dec" + str(cur),
                    nn.Linear(int(n_input_layer * factor ** cur), int(n_input_layer * factor ** (cur - 1))))
            setattr(self, "fc_bn_dec" + str(cur), nn.BatchNorm1d(int(n_input_layer * factor ** (cur - 1))))
        setattr(self, "fc_dec1",
                nn.Linear(int(n_input_layer * factor), int(n_input_layer)))

    def decode(self, z):
        if type(z)==tuple:
            z=z[0]
        h = getattr(self, "fc_bn_dec_l")(F.relu(getattr(self, "fc_dec_l")(z)))
        for cur in np.arange(self.n_reduction_layers, 1, -1):
            h = getattr(self, "fc_bn_dec" + str(cur))(F.relu(getattr(self, "fc_dec" + str(cur))(h)))

        h = F.sigmoid(getattr(self, "fc_dec1")(h))

        return h

    def forward(self, input):
        z, mu, logvar = input

        decoded = self.decode(z)
        return decoded, z, mu, logvar


class Classifier(nn.Module):

    def __init__(self, factor=1.0, n_input_layer=100,
                 n_classes=100, n_layers=2, n_reduction_layer=2, is_z=True):
        super(Classifier, self).__init__()
        self.n_reduction_layers = n_layers
        self.n_classes = n_classes
        self.n_latent_layer=n_input_layer *(1 if is_z else 2)
        self.is_z=is_z

        for cur in np.arange(1, n_layers + 1):
            setattr(self, "fc_cls" + str(cur),
                    nn.Linear(int(n_input_layer * factor ** (cur - 1)), int(n_input_layer * factor ** cur)))
            setattr(self, "fc_bn_cls" + str(cur), nn.BatchNorm1d(int(n_input_layer * factor ** cur)))




        setattr(self, "fc_cls_reduction",
                nn.Linear(int(n_input_layer * factor ** (cur)), n_reduction_layer))
        setattr(self, "fc_bn_cls_reduction", nn.BatchNorm1d(n_reduction_layer))

        self.fc_cls_out = nn.Linear(n_reduction_layer, n_classes)

    def classify(self, mu_logvar):
        h = mu_logvar
        for cur in np.arange(1, self.n_reduction_layers + 1):
            h = getattr(self, "fc_bn_cls" + str(cur))(F.relu(getattr(self, "fc_cls" + str(cur))(h)))

        h = getattr(self, "fc_bn_cls_reduction")(F.relu(getattr(self, "fc_cls_reduction")(h)))

        cls = self.fc_cls_out(h)
        return cls , h

    def forward(self, input):

        if type(input) is tuple:
            z, mu , logvar, _  = input
            z=torch.cat((mu,logvar), dim=1)
        else:
            z = input

        dummy=torch.Tensor([])
        labels_hat, reduction_layer = self.classify(z)
        return labels_hat, dummy, dummy, reduction_layer

        self.fc_cls_out = nn.Linear(n_reduction_layer, n_classes)

    def classify(self, mu_logvar):
        h = mu_logvar
        for cur in np.arange(1, self.n_reduction_layers + 1):
            h = getattr(self, "fc_bn_cls" + str(cur))(F.relu(getattr(self, "fc_cls" + str(cur))(h)))

        h = getattr(self, "fc_bn_cls_reduction")(F.relu(getattr(self, "fc_cls_reduction")(h)))

        cls = self.fc_cls_out(h)
        return cls , h

    def forward(self, input):

        if type(input) is tuple:
            z, mu , logvar, _  = input
            input= z if self.is_z else torch.cat((mu,logvar), dim=1)


        dummy=torch.Tensor([])
        labels_hat, reduction_layer = self.classify(input)
        return labels_hat, input, dummy, (input if self.is_z else reduction_layer)

