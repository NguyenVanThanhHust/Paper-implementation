import os
import numpy as np
from pysmiles import read_smiles

G = read_smiles("CN(C)C(=N)N=C(N)N", explicit_hydrogen=True)
print(G.nodes(data='element'))
print(G.edges)

import random
import matplotlib.pyplot as plt
from pysmiles import read_smiles
import pandas as pd
import logging
from tqdm import tqdm
import torch
from torch.nn import Sequential as Seq, Linear, ReLU, CrossEntropyLoss
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_geometric.data import Data

logging.getLogger('pysmiles').setLevel(logging.CRITICAL) # Anything higher than warning

# Load the data
df = pd.read_csv('curated-solubility-dataset.csv')
X_smiles = list(df['SMILES'])
Y = np.asarray(df['Solubility'])

#list of all elements in the dataset, which I've precomputed
elements = ['K', 'Y', 'V', 'Sm', 'Dy', 'In', 'Lu', 'Hg', 'Co', 'Mg',
'Cu', 'Rh', 'Hf', 'O', 'As', 'Ge', 'Au', 'Mo', 'Br', 'Ce', 
 'Zr', 'Ag', 'Ba', 'N', 'Cr', 'Sr', 'Fe', 'Gd', 'I', 'Al', 
 'B', 'Se', 'Pr', 'Te', 'Cd', 'Pd', 'Si', 'Zn', 'Pb', 'Sn', 
 'Cl', 'Mn', 'Cs', 'Na', 'S', 'Ti', 'Ni', 'Ru', 'Ca', 'Nd', 
 'W', 'H', 'Li', 'Sb', 'Bi', 'La', 'Pt', 'Nb', 'P', 'F', 'C']

print(X.shape)
print(Y.shape)

# convert element to a one-hot vector of dimension len(elements):
def convert_to_one_host(elements):
    output = []
    for idx, elem in enumerate(elements):
        v = np.zeros(len(elements))
        v[idx] = 1.0
        output.append(v)
    return output

#convert solubility value to one-hot class vector
def val_to_class(val):
    if val < -3.65: #insoluble
        return [1, 0, 0]
    elif val < -1.69: #slightly soluble
        return [0, 1, 0]
    else: #soluble
        return [0, 0, 1]

#process SMILES strings into graphs
nodes = []
edge_index = []
for smiles in tqdm(X_smiles):
    try:
        G = read_smiles(smiles, explicit_hydrogen=True)
        feature = element_to_onehot(np.asarray(G.nodes(data=’element’))[:, 1])
        edges = np.asarray(G.edges)
        index = np.asarray([edges[:,0], edges[:,1]])
        nodes.append(feature)
        edge_index.append(index)
    except:
        pass 

#Generate data objects
data = list()
# process graphs into torch geometric data objects
for idx, node in enumrate(nodes):
    x = torch.tensor(node, dtype=torch.float)
    edges = torch.tensor(edge_index[idx], dtype=torch.long)
    y = torch.tensor([val_to_class(Y[idx])], dtype=torch.float)
    data.append(Data(x=x, edge_index = edges, y=y))

random.shuffle(data)
train = data[:int(len(data)*0.8)]
test = data[int(len(data)*0.8):]

# Define network
class Network(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(61, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 32)
        self.lin1 = Linear(32, 16)
        self.lin2 = Linear(16, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.conv4(x, edge_index)
        x = F.relu(x)

        x = torch.sum(x, dim=0)
        x = self.lin1(x)
        x = F.relu(x)

        x = self.lin2(x)
        return x


