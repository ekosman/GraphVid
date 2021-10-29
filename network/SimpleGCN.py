import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, flop_count
from torch_geometric.nn import GCNConv, global_mean_pool, SplineConv, GATConv
import torch.nn as nn

from network.EdgeAttrGCN import EdgeAtrrGCNConv


class SimpleGCN(nn.Module):
    def __init__(self, num_node_features, hidden_size, num_classes=10):
        super(SimpleGCN, self).__init__()
        self.enc1 = nn.Linear(num_node_features, 128)
        self.enc2 = nn.Linear(128, 256)
        self.conv1 = EdgeAtrrGCNConv(256, hidden_size, edge_dim=1)
        self.conv2 = EdgeAtrrGCNConv(hidden_size, hidden_size, edge_dim=1)
        self.conv3 = EdgeAtrrGCNConv(hidden_size, hidden_size, edge_dim=1)
        self.conv4 = EdgeAtrrGCNConv(hidden_size, hidden_size, edge_dim=1)

        self.lin1 = nn.Linear(hidden_size, num_classes)

    def forward(self, data, flops=False):
        elu = nn.ELU()
        sum = 0
        x, edge_index, edge_attr, edge_type = data.x, data.edge_index, data.edge_attr, data.edge_type
        # edge_type = edge_type.to(torch.long)
        # edge_index = edge_index.to(torch.long)
        x = x.float()

        if flops:
            sum += FlopCountAnalysis(self.enc1, (x,)).total()
        x = self.enc1(x)

        if flops:
            sum += FlopCountAnalysis(elu, (x,)).total()

        if flops:
            sum += FlopCountAnalysis(self.enc2, (x,)).total()
        x = self.enc2(x)

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


class SimpleGAT(nn.Module):
    def __init__(self, num_node_features, hidden_size, num_classes=10):
        super(SimpleGAT, self).__init__()
        self.enc1 = nn.Linear(num_node_features, 128)
        self.enc2 = nn.Linear(128, 256)
        self.conv1 = GATConv(256, hidden_size, heads=4, concat=False)
        self.conv2 = GATConv(hidden_size, hidden_size, heads=4, concat=False)
        self.conv3 = GATConv(hidden_size, hidden_size, heads=4, concat=False)
        self.conv4 = GATConv(hidden_size, hidden_size, heads=4, concat=False)

        self.lin1 = nn.Linear(hidden_size, num_classes)

    def forward(self, data, flops=False):
        elu = nn.ELU()
        sum = 0
        x, edge_index, edge_attr, edge_type = data.x, data.edge_index, data.edge_attr, data.edge_type
        # edge_type = edge_type.to(torch.long)
        # edge_index = edge_index.to(torch.long)
        x = x.float()

        if flops:
            sum += FlopCountAnalysis(self.enc1, (x,)).total()
        x = self.enc1(x)

        if flops:
            sum += FlopCountAnalysis(elu, (x,)).total()

        if flops:
            sum += FlopCountAnalysis(self.enc2, (x,)).total()
        x = self.enc2(x)

        if flops:
            sum += FlopCountAnalysis(self.conv1, (x, edge_index)).total()
        x = self.conv1(x, edge_index)

        x = F.elu(x)
        if flops:
            sum += FlopCountAnalysis(elu, (x, )).total()

        if flops:
            sum += FlopCountAnalysis(self.conv2, (x, edge_index)).total()
        x = self.conv2(x, edge_index)

        x = F.elu(x)
        if flops:
            sum += FlopCountAnalysis(elu, x).total()

        if flops:
            sum += FlopCountAnalysis(self.conv3, (x, edge_index)).total()
        x = self.conv3(x, edge_index)

        x = F.elu(x)
        if flops:
            sum += FlopCountAnalysis(elu, x).total()

        if flops:
            sum += FlopCountAnalysis(self.conv4, (x, edge_index)).total()
        x = self.conv4(x, edge_index)

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
