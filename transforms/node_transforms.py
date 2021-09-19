import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes, subgraph

from transforms.create_superpixels_flow_graph import SPATIAL


class AdditiveGaussianNodeNoise:
    def __init__(self, sigma=1):
        """
        :param sigma: standard deviation of the added normal noise
        """
        self.sigma = sigma

    def __call__(self, g):
        g.x += torch.randn_like(g.x) * self.sigma
        return g


class RandomNodeRemoval:
    def __init__(self, p=0.8):
        """
        :param p: the probability of keeping a node
        """
        self.p = p

    def __call__(self, g):
        n_nodes = len(g.x)
        keep_nodes = torch.tensor(sorted(np.random.choice(n_nodes, int(n_nodes * self.p), replace=False)))
        edge_index, edge_attr = subgraph(keep_nodes, g.edge_index, g.edge_attr, relabel_nodes=True, num_nodes=n_nodes)
        g.x = g.x[keep_nodes]
        g.edge_index = edge_index
        g.edge_attr = edge_attr
        return g


if __name__ == '__main__':
    n_edges = 4
    n_nodes = 10
    g = Data(x=torch.randn(n_nodes, 5), edge_index=torch.randint(0, n_nodes, size=(2, n_edges)), edge_attr=torch.randn(n_edges), edge_type=torch.tensor([SPATIAL] * n_edges))
    print("Nodes before:")
    print(len(g.x))

    transform = RandomNodeRemoval(0.1)
    g = transform(g)
    print(len(g.x))
    print("Edges after:")
    print(g.x)
