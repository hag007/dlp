import os

import numpy as np
import torch
from torch import tensor
from torch.utils.data.sampler import SubsetRandomSampler

import constants_cmap


###########################################################################################

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_names, data_type):

        self.X=[]
        self.y=[]
        self.y_names=[]
        self.y_unique_names=[]
        path_to_cache=os.path.join(constants_cmap.CACHE_GLOBAL_DIR, "datasets", data_type)

        if os.path.exists(os.path.join(path_to_cache)):
            for i, name in enumerate(dataset_names):
                # if i>3 or data_type=="icgc": continue
                self.X.append(np.load(os.path.join(path_to_cache, name, "X.npy")))
                self.y.append([i for a in np.arange(self.X[-1].shape[0])]) #
                # self.y.append(np.load(os.path.join(path_to_cache, name, "y.npy")))
                self.y_names.append(np.load(os.path.join(path_to_cache, name, "y_names.npy")))
                self.y_unique_names.append(name)

        self.X=tensor(np.vstack(self.X)).float()
        self.y=tensor(np.hstack(self.y)).long()
        self.y_names=np.hstack(self.y_names)


    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]


class DatasetMask(torch.utils.data.Dataset):
    def __init__(self, dataset_names, data_type, filter_func=lambda a: True):

        self.X=[]
        self.y=[]
        self.y_names=[]
        self.filter_func=filter_func
        self.y_unique_names=[]

        path_to_cache=os.path.join(constants_cmap.CACHE_GLOBAL_DIR, "datasets", data_type)

        if os.path.exists(os.path.join(path_to_cache)):
            for i, name in enumerate(dataset_names):
                self.X.append(np.load(os.path.join(path_to_cache, name, "X.npy")))
                # self.y.append(np.load(os.path.join(path_to_cache, name, "y.npy")))
                self.y.append(np.array([i for a in np.arange(self.X[-1].shape[0])]))
                self.y_names.append(np.load(os.path.join(path_to_cache, name, "y_names.npy")))
                self.y_unique_names.append(name)

        self.X=tensor(np.vstack(self.X)).float()
        self.y=tensor(np.hstack(self.y)).long()


    def __getitem__(self, index):
        return self.X[index], (self.y[index] if self.filter_func(index)  else tensor(-1).long())

    def __len__(self):
        return self.X.shape[0]

class DataLoader(object):
    def __init__(self, dataset, validation_fraction, test_fraction, shuffle=True):

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        val_split = int(np.floor((1 - validation_fraction) * (1 - test_fraction) * dataset_size))
        tst_split = int(np.floor((1 - test_fraction) * dataset_size))
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(indices)
        train_indices, valid_indices, test_indices = indices[:val_split], indices[val_split: tst_split], indices[tst_split:]

        self.dataset = dataset
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.valid_sampler = SubsetRandomSampler(valid_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)

    def train_loader(self, batch_size=100):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=batch_size, sampler=self.train_sampler)

        return train_loader

    def valid_loader(self, batch_size=None):
        if batch_size is None:
            batch_size=self.dataset.__len__()
        test_loader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=batch_size, sampler=self.valid_sampler)

        return test_loader

    def test_loader(self, batch_size=None):
        if batch_size is None:
            batch_size=self.dataset.__len__()
        test_loader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=batch_size, sampler=self.test_sampler)

        return test_loader

    def loaders(self, batch_size):
        train_loader = self.train_loader(batch_size)
        valid_loader = self.valid_loader(batch_size)
        test_loader = self.test_loader(batch_size)

        return train_loader, valid_loader, test_loader

