import torch as th
from collections import deque
import random
import numpy as np
import pickle
# import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class SingleLayerGNN(th.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.linear = th.nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, edge_index):
        x = x.to(th.float)
        edge_index = edge_index.to(th.int64)

        # Apply GNN to the subgraph
        embedding = F.relu(th.cat((x[0], self.conv(x, edge_index)[0]), dim=0))

        # Get the AP assignment prediction
        predicted_AP_assignment = self.linear(embedding)

        return predicted_AP_assignment


class SampleBuffer(object):

    def __init__(self, batch_size, max_size=10000000):
        self.storage = deque(maxlen=max_size)
        self.batch_size = batch_size

    def add(self, transition):
        self.storage.append(transition)

    def sample(self):
        minibatch = random.sample(self.storage, self.batch_size)

    def save(self, filename):
        # Save using pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.storage, f)

    def load(self, filename):
        # Load using pickle
        with open(filename, 'rb') as f:
            self.storage = pickle.load(f)


# Create a custom dataset class
class MyGraphDataset(InMemoryDataset):

    def __init__(self, root, graphs = list([])):
        self.graphs = graphs
        super().__init__(root)

    def buffers2dataset(self, buffers, filepath):
        # run over the buffers in the list
        for buffer in buffers:
            # run over the elements in the buffer
            for sample in buffer.storage:
                # convert the matrix of channel gains to a graph
                graph = get_star_graph(sample[0].T, sample[1])
                # Append the graph and its label to the dataset
                self.graphs.append(graph)
        th.save(self.graphs, filepath+'/AP_training_Dataset.pt')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        """
        Retrieve a graph by index.
        """
        return self.graphs[idx]


def get_star_graph(channelGain_matrix, AP_assignment):
    ''''
    This function takes the features matrix and returns the star graph.
    The features matrix is a tensor of size (N, M) where N is the number of nodes (same as the number of relevant UEs)
    and M is the number of features (same as the number of potential APs).
    The star graph is a DGL graph with N nodes and N-1 edges.
    '''

    # bring the last row of the feature matrix to the first row
    featuresMatrix = th.tensor(np.roll(channelGain_matrix, 1, axis=0), dtype=th.cfloat)

    N = featuresMatrix.shape[0]
    edge_list = th.stack((th.tensor(range(1, N)), th.zeros((N-1))))

    return Data(x=featuresMatrix, edge_index=edge_list, y=th.tensor(AP_assignment))


def get_AP2UE_edges(D):
    ''''
    This function takes the AP-UE assignment matrix and returns the list of edges.
    Every row in the tensor UE2AP_edge_list represents an edge in the graph.
    The first column is the AP index and the second column is the UE index.
    '''

    AP2UE_edges = th.tensor(np.transpose(np.nonzero(D)))

    return AP2UE_edges


def get_Pilot2UE_edges(pilotIndex):
    ''''
    This function takes the pilot allocation and returns the list of edges.
    Every row in the tensor UE2AP_edge_list represents an edge in the graph.
    The first column is the AP index and the second column is the UE index.
    '''

    Pilot2UE_edges = th.zeros((len(pilotIndex), 2))

    for idx in range(len(pilotIndex)):
        Pilot2UE_edges[idx, 0] = pilotIndex[idx]
        Pilot2UE_edges[idx, 1] = idx

    return Pilot2UE_edges


def get_oneHot_bestPilot(best_pilot_sample, tau_p):
    ''''
    This function takes the best pilot allocation and returns the one-hot encoding of the best pilot.
    '''

    one_hot_best_pilot = th.zeros((tau_p))

    one_hot_best_pilot[int(best_pilot_sample)] = 1

    return one_hot_best_pilot
