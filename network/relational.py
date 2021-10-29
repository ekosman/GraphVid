import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
from torch_geometric.nn import global_mean_pool, RGCNConv

from network.EdgeAtrrRGCN import EdgeAtrrRGCN


class DynamicGCN(nn.Module):
    def __init__(self, num_node_features, hidden_size, num_classes=10):
        super(DynamicGCN, self).__init__()
        self.conv1 = RGCNConv(num_node_features, hidden_size, num_relations=2)
        self.conv2 = RGCNConv(hidden_size, hidden_size, num_relations=2)
        self.conv3 = RGCNConv(hidden_size, hidden_size, num_relations=2)

        self.lin1 = nn.Linear(hidden_size, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, edge_type = data.x, data.edge_index, data.edge_attr, data.edge_type
        # edge_type = edge_type.to(torch.long)
        # edge_index = edge_index.to(torch.long)
        x = x.float()
        x = self.conv1(x, edge_index, edge_type)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = F.elu(x)
        x = self.conv3(x, edge_index, edge_type)
        x = F.elu(x)

        # 2. Readout layer
        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin1(x)
        return x


class DynamicGCNWEdgeAttrs(nn.Module):
    def __init__(self, num_node_features, hidden_size, num_classes=10):
        super(DynamicGCNWEdgeAttrs, self).__init__()
        self.enc1 = nn.Linear(num_node_features, 128)
        self.enc2 = nn.Linear(128, 256)
        self.conv1 = EdgeAtrrRGCN(256, hidden_size, num_relations=2, edge_dim=1)
        self.conv2 = EdgeAtrrRGCN(hidden_size, hidden_size, num_relations=2, edge_dim=1)
        self.conv3 = EdgeAtrrRGCN(hidden_size, hidden_size, num_relations=2, edge_dim=1)
        self.conv4 = EdgeAtrrRGCN(hidden_size, hidden_size, num_relations=2, edge_dim=1)
        # self.conv5 = EdgeAtrrRGCN(hidden_size, hidden_size, num_relations=2, edge_dim=1)
        # self.conv6 = EdgeAtrrRGCN(hidden_size, hidden_size, num_relations=2, edge_dim=1)

        self.lin1 = nn.Linear(hidden_size, num_classes)

    def forward(self, data, flops=False):
        elu = nn.ELU()
        sum = 0
        x, edge_index, edge_attr, edge_type = data.x, data.edge_index, data.edge_attr, data.edge_type
        # edge_type = edge_type.to(torch.long)
        # edge_index = edge_index.to(torch.long)
        x = x.float()

        if flops:
            sum += FlopCountAnalysis(self.enc1, (x, )).total()
        x = self.enc1(x)

        if flops:
            sum += FlopCountAnalysis(elu, (x, )).total()

        if flops:
            sum += FlopCountAnalysis(self.enc2, (x,)).total()
        x = self.enc2(x)

        if flops:
            sum += FlopCountAnalysis(elu, (x,)).total()

        if flops:
            sum += FlopCountAnalysis(self.conv1, (x, edge_index, edge_type, edge_attr)).total()
        x = self.conv1(x, edge_index, edge_type, edge_attr)

        x = F.elu(x)
        if flops:
            sum += FlopCountAnalysis(elu, (x, )).total()

        if flops:
            sum += FlopCountAnalysis(self.conv2, (x, edge_index, edge_type, edge_attr)).total()
        x = self.conv2(x, edge_index, edge_type, edge_attr)

        x = F.elu(x)
        if flops:
            sum += FlopCountAnalysis(elu, x).total()

        if flops:
            sum += FlopCountAnalysis(self.conv3, (x, edge_index, edge_type, edge_attr)).total()
        x = self.conv3(x, edge_index, edge_type, edge_attr)

        x = F.elu(x)
        if flops:
            sum += FlopCountAnalysis(elu, x).total()

        if flops:
            sum += FlopCountAnalysis(self.conv4, (x, edge_index, edge_type, edge_attr)).total()
        x = self.conv4(x, edge_index, edge_type, edge_attr)

        x = F.elu(x)
        if flops:
            sum += FlopCountAnalysis(elu, x).total()

        # 2. Readout layer
        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training)

        if flops:
            sum += FlopCountAnalysis(self.lin1, x).total()
        x = self.lin1(x)

        if flops:
            return x, sum

        return x
