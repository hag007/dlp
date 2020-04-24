import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import tensor
import pandas as pd
import os
from torch.utils.data.sampler import SubsetRandomSampler
import constants
from scipy.stats import zscore, rankdata
import shutil

    ###########################################################################################

class CMAPDataset(torch.utils.data.Dataset):
    def __init__(self, genes=None,genes_name=None, dataset_names=None, data_type=None):

        if dataset_names is None:
            dataset_names=[a for i, a in enumerate(constants.DATASETS_NAMES) if constants.DATASETS_INCLUDED[i]]

        if data_type is None:
            data_type=constants.DATA_TYPE

        self.X=[]
        self.y=[]
        self.y_names=[]
        self.y_f=[]
        self.y_f_names=[]
        path_to_cache=os.path.join(constants.CACHE_GLOBAL_DIR, "datasets", data_type)

        if os.path.exists(os.path.join(path_to_cache)):
            for i, name in enumerate(dataset_names):
                self.X.append(np.load(os.path.join(path_to_cache, name, "X.npy")))
                # self.y.append([i for a in np.arange(self.X[-1].shape[0])]) #
                self.y.append(np.load(os.path.join(path_to_cache, name, "y.npy")))
                self.y_names.append(np.load(os.path.join(path_to_cache, name, "y_names.npy")))
                self.y_f.append(np.load(os.path.join(path_to_cache, name, "y_f.npy")))
                self.y_f_names.append(np.load(os.path.join(path_to_cache, name, "y_f_names.npy")))

        self.X=tensor(np.vstack(self.X)).float()
        self.y=tensor(np.hstack(self.y)).long()
        self.y_names=np.hstack(self.y_names)
        self.y_f=tensor(np.hstack(self.y_f)).long()
        self.y_f_names=np.hstack(self.y_f_names)


    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]


class CMAPDatasetMask(torch.utils.data.Dataset):
    def __init__(self, genes=None,genes_name=None, filter_func=lambda a: True, dataset_names=None, data_type=None):
        if dataset_names is None:
            dataset_names=[a for i, a in enumerate(constants.DATASETS_NAMES) if constants.DATASETS_INCLUDED[i]]

        if data_type is None:
            data_type=constants.DATA_TYPE

        self.X=[]
        self.y=[]
        self.y_names=[]
        self.y_f=[]
        self.y_f_names=[]
        self.filter_func=filter_func

        path_to_cache=os.path.join(constants.CACHE_GLOBAL_DIR, "datasets",data_type)

        if os.path.exists(os.path.join(path_to_cache)):
            for i, name in enumerate(dataset_names):
                self.X.append(np.load(os.path.join(path_to_cache, name, "X.npy")))
                self.y.append(np.load(os.path.join(path_to_cache, name, "y.npy")))
                # self.y.append(np.array([i for a in np.arange(self.X[-1].shape[0])]))
                self.y_names.append(np.load(os.path.join(path_to_cache, name, "y_names.npy")))
                self.y_f.append(np.load(os.path.join(path_to_cache, name, "y_f.npy")))
                self.y_f_names.append(np.load(os.path.join(path_to_cache, name, "y_f_names.npy")))

        self.X=tensor(np.vstack(self.X)).float()
        self.y=tensor(np.hstack(self.y)).long()
        self.y_f=tensor(np.hstack(self.y_f)).long()
        self.y_f_names=np.hstack(self.y_f_names)


    def __getitem__(self, index):
        return self.X[index], (self.y[index] if self.filter_func(index)  else tensor(-1).long())

    def __len__(self):
        return self.X.shape[0]

class CMAPDataLoader(object):
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


#################################################################################################
# class CMAPDataLoader(object):
#     def __init__(self, dataset, validation_split, shuffle=True):
#         dataset_size = len(dataset)
#         indices = list(range(dataset_size))
#         split = int(np.floor(validation_split * dataset_size))
#         if shuffle:
#             np.random.seed(42)
#             np.random.shuffle(indices)
#         train_indices, valid_indices = indices[split:], indices[: split]
#
#         self.dataset = dataset
#         self.train_sampler = SubsetRandomSampler(train_indices)
#         self.valid_sampler = SubsetRandomSampler(valid_indices)
#
#     def train_loader(self, batch_size):
#         train_loader = torch.utils.data.DataLoader(
#             dataset=self.dataset, batch_size=batch_size, sampler=self.train_sampler)
#
#         return train_loader
#
#     def test_loader(self, batch_size):
#         test_loader = torch.utils.data.DataLoader(
#             dataset=self.dataset, batch_size=batch_size, sampler=self.valid_sampler)
#
#         return test_loader
#
#     def loaders(self, batch_size):
#         train_loader = self.train_loader(batch_size)
#         test_loader = self.test_loader(batch_size)
#
#         return train_loader, test_loader

#
# ##############################################
#
#
# class CMAP(object):
#     def __init__(self, path_data="datasets/CMAP/", datasets=datasets, replications=1):
#
#
#
#     def __iter__(self):
#         all_datasets=[pd.read_csv(ds, sep='\t', index_col=0).drop(
#                 ["pr_gene_symbol", "pr_gene_symbol.1"]).T for ds in datasets]
#         self.df_X_all = pd.concat(all_datasets, axis=1)
#         self.df_y_all = [datasets[all_datasets.index(a)] for a in all_datasets for b in a]
#         self.df_y_all_names = [datasets[all_datasets.index(a)].split("_")[1] for a in self.df_y_all]
#
#         X, y, = self.df_X_all, self.df_y_all
#         yield (X, y)
#
#     def get_train_valid_test(self):
#
#             df_data_case = pd.read_csv(self.data_case, sep='\t', index_col=0).drop(
#                 ["pr_gene_symbol", "pr_gene_symbol.1"],
#                 axis=1).T
#             df_data_case.T = 1
#             df_data_control = pd.read_csv(self.data_control, sep='\t', index_col=0).drop(
#                 ["pr_gene_symbol", "pr_gene_symbol.1"], axis=1).T
#             df_data_control.T = 0
#             df_data = pd.concat([df_data_case, df_data_control], axis=0)
#
#             bg_genes = open(os.path.join(self.path_data, "bg_genes.txt")).read().split("\n")
#             affecting_genes = open(os.path.join(self.path_data, "affecting_genes.txt")).read().split("\n")
#             if affecting_genes[0] == '':
#                 affecting_genes = []
#             outcome_genes = open(os.path.join(self.path_data, "outcome_genes.txt")).read().split("\n")
#
#             t, y, = df_data.loc[:, ['T']], df_data.loc[:, outcome_genes]
#             x = df_data.loc[:, affecting_genes+bg_genes]
#
#             idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
#             itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
#             train = (x.iloc[itr].values, t.iloc[itr].values, y.iloc[itr].values)
#             valid = (x.iloc[iva].values, t.iloc[iva].values, y.iloc[iva].values)
#             test = (x.iloc[ite].values, t.iloc[ite].values, y.iloc[ite].values)
#             yield train, valid, test, self.contfeats, self.binfeats