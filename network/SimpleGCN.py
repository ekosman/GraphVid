import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, SplineConv, GATConv
import torch.nn as nn

from network.EdgeAttrGCN import EdgeAtrrGCNConv


class SimpleGCN(nn.Module):
    def __init__(self, num_node_features, hidden_size, num_classes=10):
        super(SimpleGCN, self).__init__()
        self.conv1 = EdgeAtrrGCNConv(num_node_features, hidden_size, edge_dim=1)
        self.conv2 = EdgeAtrrGCNConv(hidden_size, hidden_size, edge_dim=1)
        self.conv3 = EdgeAtrrGCNConv(hidden_size, hidden_size, edge_dim=1)
        self.conv4 = EdgeAtrrGCNConv(hidden_size, hidden_size, edge_dim=1)

        self.lin1 = nn.Linear(hidden_size, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, edge_type = data.x, data.edge_index, data.edge_attr, data.edge_type
        # edge_type = edge_type.to(torch.long)
        # edge_index = edge_index.to(torch.long)
        x = x.float()
        x = self.conv1(x, edge_index, edge_type, edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_type, edge_attr)
        x = F.elu(x)
        x = self.conv3(x, edge_index, edge_type, edge_attr)
        x = F.elu(x)
        x = self.conv4(x, edge_index, edge_type, edge_attr)
        x = F.elu(x)
        # x = self.conv5(x, edge_index, edge_type, edge_attr)
        # x = F.elu(x)
        # x = self.conv6(x, edge_index, edge_type, edge_attr)
        # x = F.elu(x)

        # 2. Readout layer
        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin1(x)
        return x
