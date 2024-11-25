import torch as th
from collections import deque
import random
import numpy as np
import pickle
import dgl
from dgl.data import DGLDataset


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


# Step 1: Create a custom dataset class
class MyGraphDataset(DGLDataset):

    def __init__(self):
        super().__init__(name='my_graph_dataset')
        self.graphs = []
        self.labels = []

    def buffers2dataset(self, buffers):
        # run over the buffers in the list
        for buffer in buffers:
            # run over the elements in the buffer
            for sample in buffer.storage:
                # convert the matrix of channel gains to a graph
                graph = get_star_graph(sample[0].T)
                # Store the AP assignment as a tensor-like label for the graph
                label = th.tensor(sample[1])
                # Append the graph and its label to the dataset
                self.graphs.append(graph)
                self.labels.append(label)

    def __getitem__(self, idx):
        # Return the graph and its label
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        # Return the total number of graphs
        return len(self.graphs)
        return len(self.graphs)


def get_star_graph(featuresMatrix):
    ''''
    This function takes the features matrix and returns the star graph.
    The features matrix is a tensor of size (N, M) where N is the number of nodes (same as the number of relevant UEs)
    and M is the number of features (same as the number of potential APs).
    The star graph is a DGL graph with N nodes and N-1 edges.
    '''

    # bring the last row of the feature matrix to the first row
    featuresMatrix = np.roll(featuresMatrix, 1, axis=0)

    N = featuresMatrix.shape[0]

    # Create the star graph
    G = dgl.DGLGraph()
    G.add_nodes(N)
    G.add_edges(th.tensor(range(1, N)), 0)

    # Assign the features to the nodes
    G.ndata['feat'] = th.tensor(featuresMatrix, dtype=th.cfloat)

    return G





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
