import os

import numpy as np
import scipy.sparse as sp
import torch
from six.moves import cPickle as pickle
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops
from preprocessing import load_data
import pandas as pd
from collections import Counter


df = pd.read_csv('./data/convert/baron3_HVG_1200.csv', header=None)

# Convert the DataFrame to a NumPy array
data_array = df.values
data_array_trimmed = data_array[1:, 1:]
data_array_trimmed = data_array_trimmed.astype(float).T
print(data_array_trimmed.shape)
npy_file_path = "./data/singlecell/baron3-x.npy"
np.save(npy_file_path, data_array_trimmed)

#labels
labels = pd.read_csv('./data/convert/mod-Human_kidney_Y.csv') 
labels['name'] = labels['name'] + 1
labels['name'] = labels['name'].astype(float)
y=labels['name']
print(y.value_counts())
y_binary = y.copy()
#adjust based on target label
y_binary[y_binary != 5.0] = 0.0 #
y_binary[y_binary == 5.0] = 1.0 #
np.save('./data/singlecell/human_kidney-y5.npy', y_binary.to_numpy()) #

indices = y[y == 1.0].index

#edges
with open("./data/convert/baron3_HVG_1200_KNN_k5_d10.txt", 'r') as f:
    edge_list = [list(map(int, line.strip().split())) for line in f]
edges = torch.tensor(edge_list, dtype=torch.long)
edges = edges.t().contiguous()
print(edges.shape)
np.save('./data/singlecell/baron3-edge.npy',edges)

