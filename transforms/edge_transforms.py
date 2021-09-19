import torch
import numpy as np
from torch_geometric.data import Data

from transforms.create_superpixels_flow_graph import SPATIAL, TEMPORAL


class AdditiveGaussianEdgeNoise:
    def __init__(self, sigma=0.05):
        """
        :param sigma: standard deviation of the added normal noise
        """
        self.sigma = sigma

    def __call__(self, g):
        g.edge_attr += torch.randn_like(g.edge_attr) * self.sigma
        return g


class RandomSpatialEdgesRemoval:
    def __init__(self, p=0.8):
        """
        :param p: the probability of keeping a node
        """
        self.p = p

    def __call__(self, g):
        spatial_indices, = torch.where(g.edge_type == SPATIAL)
        temporal_indices, = torch.where(g.edge_type == TEMPORAL)
        spatial_indices = torch.tensor(np.random.choice(spatial_indices, int(len(spatial_indices) * self.p), replace=False))
        chosen_indices = torch.cat([spatial_indices, temporal_indices])
        g.edge_index = g.edge_index[:, chosen_indices]
        if g.edge_attr is not None:
            g.edge_attr = g.edge_attr[chosen_indices, ...]
        if g.edge_type is not None:
            g.edge_type = g.edge_type[chosen_indices]
        return g


if __name__ == '__main__':
    n_edges = 4
    n_nodes = 5
    g = Data(x=torch.randn(n_nodes, 5), edge_index=torch.randint(0, n_nodes, size=(2, n_edges)), edge_attr=torch.randn(n_edges), edge_type=torch.tensor([SPATIAL] * n_edges))
    print("Edges before:")
    print(g.edge_index)

    transform = RandomSpatialEdgesRemoval()
    g = transform(g)
    print("Edges after:")
    print(g.edge_index)
