import sys
import torch
import logging
import ipdb
from torch_geometric.utils import to_scipy_sparse_matrix
import torch.optim as optim
import scipy.sparse as sp
import numpy as np
from sklearn import preprocessing
from scipy.signal import (
    convolve, butter,
    lfilter, resample,
    hilbert, filtfilt, medfilt,
    savgol_filter, iirnotch
)
def make_optimizer(model, opt, lr, weight_decay, momentum):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    parameters = [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

    # weight_decay = 0.

    opt_args = dict(weight_decay=weight_decay, lr=lr)

    if opt.lower() == 'sgd':
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt.lower() == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt.lower() == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)

    return optimizer


class Optimizers(object):
    def __init__(self):
        self.optimizers = []
        self.lrs = []

    def add(self, optimizer, lr):
        self.optimizers.append(optimizer)
        self.lrs.append(lr)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def __getitem__(self, index):
        return self.optimizers[index]

    def __setitem__(self, index, value):
        self.optimizers[index] = value


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor([0.], device='cuda:0')
        self.n = torch.tensor([0.], device='cuda:0')
        return

    def update(self, val, num):
        self.sum += val * num
        self.n += num

    @property
    def avg(self):
        return self.sum / self.n


def classification_accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def set_dataset_paths(args):
    """Set default train and test path if not provided as input."""

    if not args.train_path:
        args.train_path = 'data/%s/train' % (args.dataset)

    if not args.val_path:
        if (args.dataset in ['imagenet', 'face_verification', 'emotion', 'gender'] or
            args.dataset[:3] == 'age'):
            args.val_path = 'data/%s/val' % (args.dataset)
        else:
            args.val_path = 'data/%s/test' % (args.dataset)


def set_logger(filepath):
    global logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(filepath)
        ch = logging.StreamHandler(sys.stdout)

        _format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(_format)
        ch.setFormatter(_format)

        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


class EarlyStopping(object):
    def __init__(self, patience):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        is_best, is_terminate = True, False
        if self.best_score is None:
            self.best_score = score
        elif self.best_score >= score:
            self.counter += 1
            if self.counter >= self.patience:
                is_terminate = True
            is_best = False
        else:
            self.best_score = score
            self.counter = 0
        return is_best, is_terminate

def to_sparse_tensor(data):
    """Convert edge_index to sparse matrix"""
    edge_index = data.edge_index
    sparse_mx = to_scipy_sparse_matrix(edge_index)
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if not isinstance(sparse_mx, sp.coo_matrix):
        sparse_mx = sp.coo_matrix(sparse_mx)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.LongTensor(np.array([sparse_mx.row, sparse_mx.col]))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=shape
    )

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    train_index = torch.from_numpy(np.arange(140))
    val_index   = torch.from_numpy(np.arange(140, 640))
    test_index  = torch.from_numpy(np.arange(1708, 2708))

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    # for i in range(num_classes):
    #     index = (data.y == i).nonzero().view(-1)# 数据集中，标签类别为i的索引号 规模：[Num_i]
    #     index = index[torch.randperm(index.size(0))]# 打乱上述索引号
    #     indices.append(index)# 最终是长度为num_classes的列表
    #
    # train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)#每一类取出percls_trn条数据，拼成总的训练集，每类可能取不够percls_trn
    #
    # if Flag == 0:
    #     rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    #     rest_index = rest_index[torch.randperm(rest_index.size(0))]#取剩余数据，索引打乱
    #
    #     data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    #     data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)# 剩余数据取前val_lb条做验证
    #     data.test_mask = index_to_mask(
    #         rest_index[val_lb: val_lb+1000], size=data.num_nodes)# 最后剩下的做测试
    # else:
    #     val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
    #                            for i in indices], dim=0)# 每一类都取出val_lb条做验证
    #     rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)# 再剩下的做测试
    #     rest_index = rest_index[torch.randperm(rest_index.size(0))]
    #
    #     data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    #     data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    #     data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data

def highpass_filter(signal, hpf, fs):

    hp = hpf / (fs*0.5)
    hpb, hpa = butter(5, hp, 'highpass')
    hp_signal = filtfilt(hpb, hpa, signal, axis=0)

    return hp_signal

def lowpass_filter(signal, lpf, fs):

    lp = lpf / (fs*0.5)
    lpb, lpa = butter(5, lp, 'lowpass')
    lp_signal = filtfilt(lpb, lpa, signal, axis=0)

    return lp_signal

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    data = preprocessing.minmax_scale(data)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y