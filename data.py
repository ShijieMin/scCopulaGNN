import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import random
from six.moves import cPickle as pickle
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops
from sklearn.model_selection import StratifiedShuffleSplit
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from utils import args
from imblearn.over_sampling import SMOTE, SVMSMOTE, RandomOverSampler, BorderlineSMOTE, ADASYN
from torch_geometric.utils import negative_sampling as nsampling


def to_data(x, y, adj=None, edge_index=None, train_idx=None, valid_idx=None,
            test_idx=None, train_size=1. / 10, valid_size=1. / 10):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    if edge_index is None: #we provide edge_index in our data
        assert adj is not None
        edge_index = torch.tensor(np.array(list(adj.nonzero())))
    else: #convert to a tensor type and ensure the graph is undirected
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
    edge_index = remove_self_loops(to_undirected(edge_index))[0]

    data = Data(x=x_tensor, y=y_tensor, edge_index=edge_index)
    n = data.x.size(0)
    if train_idx is not None:
        assert valid_idx is not None and test_idx is not None
        all_idx = set(list(range(n)))
        train_idx = set(train_idx)
        valid_idx = set(valid_idx)
        test_idx = all_idx.difference(train_idx.union(valid_idx))
    elif isinstance(train_size, float):
        train_size = int(n * train_size)
        valid_size = int(n * valid_size)
        test_size = n - train_size - valid_size
        train_idx = set(range(train_size))
        valid_idx = set(range(train_size, train_size + valid_size))
        test_idx = set(range(n - test_size, n))
    assert len(test_idx.intersection(train_idx.union(valid_idx))) == 0
    data.train_mask = torch.zeros(n).to(dtype=torch.bool)
    data.train_mask[list(train_idx)] = True
    data.valid_mask = torch.zeros(n).to(dtype=torch.bool)
    data.valid_mask[list(valid_idx)] = True
    data.test_mask = torch.zeros(n).to(dtype=torch.bool)
    data.test_mask[list(test_idx)] = True
    return data

def read_sota(path, seed=1):
    data_path = os.path.join(path, "singlecell")
    x = np.load(os.path.join(data_path, "sota-x.npy"))
    #change the y file based on which label we would like
    y = np.load(os.path.join(data_path, f"sota-y{args.num_class}.npy"), allow_pickle=True)
    edge_index = np.load(
        os.path.join(data_path, "sota-edge.npy"))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=8/10, random_state=seed)
    for train_idx, temp_idx in sss.split(x, y):
        break
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    for valid_idx, test_idx in sss.split(x[temp_idx], y[temp_idx]):
        break
    valid_idx = temp_idx[valid_idx]
    test_idx = temp_idx[test_idx]
    data = to_data(x, y, edge_index=edge_index, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
    return data


def read_baron3(path, seed=1):
    data_path = os.path.join(path, "singlecell")
    x = np.load(os.path.join(data_path, "baron3-x.npy"))
    y = np.load(os.path.join(data_path, f"baron3-y{args.num_class}.npy"))
    edge_index = np.load(
        os.path.join(data_path, "baron3-edge.npy"))
    rs = np.random.RandomState(seed)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=8/10, random_state=seed)
    for train_idx, temp_idx in sss.split(x, y):
        break
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    for valid_idx, test_idx in sss.split(x[temp_idx], y[temp_idx]):
        break
    valid_idx = temp_idx[valid_idx]
    test_idx = temp_idx[test_idx]
    data = to_data(x, y, edge_index=edge_index, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
    return data


def read_hk(path, seed=1):
    data_path = os.path.join(path, "singlecell")
    x = np.load(os.path.join(data_path, "human_kidney-x.npy"))
    y = np.load(os.path.join(data_path, f"human_kidney-y{args.num_class}.npy"), allow_pickle=True)

    edge_index = np.load(os.path.join(data_path, "human_kidney-edge.npy"))
    rs = np.random.RandomState(seed)
    idx = rs.permutation(len(y))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=8/10, random_state=seed)
    for train_idx, temp_idx in sss.split(x, y):
        break
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    for valid_idx, test_idx in sss.split(x[temp_idx], y[temp_idx]):
        break
    valid_idx = temp_idx[valid_idx]
    test_idx = temp_idx[test_idx]

    data = to_data(x, y, edge_index=edge_index, train_idx=train_idx,
                   valid_idx=valid_idx, test_idx=test_idx)
    print(data)
    return data


def read_tb(path, seed=1):
    data_path = os.path.join(path, "singlecell")
    x = np.load(os.path.join(data_path, "tb-x.npy"))
    y = np.load(os.path.join(data_path, f"tb-y{args.num_class}.npy"), allow_pickle=True)

    edge_index = np.load(
        os.path.join(data_path, "tb-edge.npy"))
    rs = np.random.RandomState(seed)
    idx = rs.permutation(len(y))
    train_size = int(0.8 * len(idx))
    valid_size = int(0.1 * len(idx))
    test_size = len(idx) - train_size - valid_size
    train_idx = idx[:train_size]
    valid_idx = idx[train_size:train_size + valid_size]
    test_idx = idx[train_size + valid_size:]
    data = to_data(x, y, edge_index=edge_index, train_idx=train_idx,
                   valid_idx=valid_idx, test_idx=test_idx)
    print(data)
    return data