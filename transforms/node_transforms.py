import torch
import numpy as np
from torch_geometric.utils import remove_isolated_nodes


class AdditiveGaussianNodeNoise:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, g):
        g.x += torch.randn_like(g.x) * self.sigma
        return g


class RandomNodeRemoval:
    def __init__(self, p):
        self.p = p

    def __call__(self, g):
        n_nodes = len(g.x)
        removed_nodes = np.random.choice(n_nodes, int(n_nodes * (1 - self.p)), replace=False)
        edges_from = 1 - (g.edge_index[0, :].view(-1, 1) == removed_nodes.view(1, -1)).sum(dim=1)
        edges_to = 1 - (g.edge_index[1, :].view(-1, 1) == removed_nodes.view(1, -1)).sum(dim=1)
        edges_to_keep = torch.clip(edges_from + edges_to, 0, 1)
        edge_index = g.edge_index[edges_to_keep == 1]
        edge_attr = g.edge_attr[edges_to_keep == 1]
        edge_type = g.edge_type[edges_to_keep == 1]
        edge_index, edge_attr, mask = remove_isolated_nodes(edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_nodes)
        g.x = g.x[mask]
        g.edge_index = edge_index
        g.edge_attr = edge_attr
        g.edge_type = edge_type
        return g
