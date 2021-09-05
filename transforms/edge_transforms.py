import torch
import numpy as np

from transforms.create_superpixels_flow_graph import SPATIAL, TEMPORAL


class AdditiveGaussianEdgeNoise:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, g):
        g.edge_attr += torch.randn_like(g.edge_attr) * self.sigma
        return g


class RandomSpatialEdgesRemoval:
    def __init__(self, p):
        self.p = p

    def __call__(self, g):
        spatial_indices = torch.where(g.edge_type == SPATIAL)
        temporal_indices = torch.where(g.edge_type == TEMPORAL)
        spatial_indices = np.random.choice(spatial_indices, int(len(spatial_indices) * self.p), replace=False)
        chosen_indices = torch.cat(spatial_indices, temporal_indices)
        g.edge_index = g.edge_index[:, chosen_indices]
        g.edge_attr = g.edge_attr[chosen_indices, ...]
        g.edge_type = g.edge_type[chosen_indices]
        return g