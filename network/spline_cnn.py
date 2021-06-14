import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, graclus, max_pool
import torch.nn as nn
from src.archs.monet import normalized_cut_2d

"""
Implementation from: https://github.com/rusty1s/pytorch_spline_conv
"""


class SplineCNN(torch.nn.Module):
    def __init__(self):
        super(SplineCNN, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=1, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=1, kernel_size=5)
        self.lin1 = nn.Linear(64, 128)
        self.lin2 = nn.Linear(128, 10)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x)
        x = F.elu(x)

        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(self.conv3(x, edge_index, edge_attr))
        x = self.conv4(x, edge_index, edge_attr)
        x = F.elu(self.conv5(x, edge_index, edge_attr))
        x = self.conv6(x, edge_index, edge_attr)
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)
