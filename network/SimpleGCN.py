import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, SplineConv, GATConv
import torch.nn as nn


class SimpleGCN(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(SimpleGCN, self).__init__()
        hidden_channels = 16
        self.conv1 = GATConv(num_node_features, 32)
        self.conv2 = GATConv(32, 64)
        self.conv3 = GATConv(64, 128)
        self.conv4 = GATConv(128, 256)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.conv1 = SplineConv(1, 32, dim=1, kernel_size=5)

        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 10)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = self.conv4(x, edge_index)
        x = F.elu(x)

        # 2. Readout layer
        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = self.lin2(x)
        return x
        # return F.log_softmax(x, dim=1)


