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
        # edges_from = 1 - (g.edge_index[0, :].view(-1, 1) == removed_nodes.view(1, -1)).sum(dim=1)
        # edges_to = 1 - (g.edge_index[1, :].view(-1, 1) == removed_nodes.view(1, -1)).sum(dim=1)
        # edges_to_keep = torch.clip(edges_from * edges_to, 0, 1)
        # edge_index = g.edge_index[:, edges_to_keep == 1]
        # edge_attr = g.edge_attr[edges_to_keep == 1]
        # edge_type = g.edge_type[edges_to_keep == 1]
        # edge_index, edge_attr, mask = remove_isolated_nodes(edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_nodes)
        # g.x = g.x[mask]
        # g.edge_index = edge_index
        # g.edge_attr = edge_attr
        # g.edge_type = edge_type
        edge_index, edge_attr = subgraph(keep_nodes, g.edge_index, g.edge_attr, relabel_nodes=True, num_nodes=n_nodes)
        g.x = g.x[keep_nodes]
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
