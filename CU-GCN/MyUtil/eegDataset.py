# -*- coding: utf-8 -*-
"""

"""
import ipdb
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.data import Data, Dataset, DataLoader, InMemoryDataset

import numpy as np
import scipy.io as sio
import scipy
from scipy.sparse import csr_matrix
from MyUtil.data_preprocess import EEGprocess
from MyUtil.Electrodes_62 import Electrodes
import matplotlib.pyplot as plt

import time


# %% Dataset

class SDdataset(Dataset):
    def __init__(self, args, stage, subject, session, softlabel=False, pre_adj=False):
        super().__init__()
        self.args = args
        self.session = session  # the index of session order, [1,2,3]
        self.idx = subject  # the index of trails selected as testset
        self.stage = stage  # "train" or "test"
        Process = EEGprocess(args)

        if args.train_pattern == 'SD':
            traindata, testdata, trainlabel, testlabel, adj_freq = Process.SD_process(subject, session)
        elif args.train_pattern == 'SI':
            traindata, testdata, trainlabel, testlabel, adj_freq = Process.SI_process(subject, session, stage)

        epsilon = 0.2
        if stage == 'train':
            self.x = torch.from_numpy(traindata).float()
            self.y = torch.from_numpy(trainlabel).float()
            if args.dataset == 'SEEDIV':
                self.yprob = np.zeros((trainlabel.shape[0], 4))
                if softlabel is True:
                    for i, label in enumerate(trainlabel):
                        if label[0] == 0:
                            self.yprob[i, :] = [1 - (3 * epsilon / 4), epsilon / 4, epsilon / 4, epsilon / 4]
                        elif label[0] == 1:
                            self.yprob[i, :] = [epsilon / 3, 1 - (2 * epsilon / 3), epsilon / 3, 0]
                        elif label[0] == 2:
                            self.yprob[i, :] = [epsilon / 4, epsilon / 4, 1 - (3 * epsilon / 4), epsilon / 4]
                        elif label[0] == 3:
                            self.yprob[i, :] = [epsilon / 3, 0, epsilon / 3, 1 - (2 * epsilon / 3)]

                self.yprob = torch.from_numpy(self.yprob).float()
            elif args.dataset == 'SEED':
                self.yprob = np.zeros((trainlabel.shape[0], 3))
                if softlabel is True:
                    for i, label in enumerate(trainlabel):
                        if label[0] == 0:
                            self.yprob[i, :] = [1 - (2 * epsilon / 3), 2 * epsilon / 3, 0]
                        elif label[0] == 1:
                            self.yprob[i, :] = [epsilon / 3, 1 - (2 * epsilon / 3), epsilon / 3]
                        elif label[0] == 2:
                            self.yprob[i, :] = [0, 2 * epsilon / 3, 1 - (2 * epsilon / 3)]
                self.yprob = torch.from_numpy(self.yprob).float()
        elif stage == 'test':
            self.x = torch.from_numpy(testdata).float()
            self.y = torch.from_numpy(testlabel).float()

        if pre_adj is True:
            tmp = sio.loadmat(args.data_path.format(dataset=args.dataset) + 'initial_A_62x62.mat')
            self.adj = torch.from_numpy(tmp['initial_A']).float()
        else:
            elec = Electrodes(add_global_connections=True, expand_3d=False)
            self.adj = torch.from_numpy(elec.adjacency_matrix).float()
            # self.adj = torch.from_numpy(adj_freq).type(torch.FloatTensor)

        row = torch.from_numpy(np.arange(62).repeat(62))
        col = torch.from_numpy(np.tile(np.arange(62), 62))
        self.edge_index = torch.stack((row, col), 0)

    def __len__(self):
        return np.size(self.y, 0)

    def __getitem__(self, idx):
        if self.stage == 'train':
            return Data(x=self.x[idx, :, :], y=self.y[idx, :].squeeze(),
                        yprob=self.yprob[idx, :].reshape(1, self.yprob.size(1)),
                        edge_index=self.edge_index)
        elif self.stage == 'test':
            return Data(x=self.x[idx, :, :], y=self.y[idx, :].squeeze(), edge_index=self.edge_index)

def train_loader(args, stage, index, session, num_workers=8, pin_memory=True):
    train_dataset = SDdataset(args, stage, index, session)

    return DataLoader(train_dataset,
                      batch_size=args.train_batch_size, shuffle=True, sampler=None,
                      num_workers=num_workers, pin_memory=pin_memory)


def test_loader(args, stage, index, session, num_workers=4, pin_memory=False):
    test_dataset = SDdataset(args, stage, index, session)

    return DataLoader(test_dataset,
                      batch_size=args.val_batch_size, shuffle=False, sampler=None,
                      num_workers=num_workers, pin_memory=pin_memory)
