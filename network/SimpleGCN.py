import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, SplineConv, GATConv
import torch.nn as nn


class SimpleGCN(nn.Module):
    def __init__(self, num_node_features, hidden_size, num_classes=10):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, num_classes)
        # self.conv4 = GATConv(1024, 1024, heads=4, concat=False)

        self.lin1 = nn.Linear(128, num_classes)
        # self.lin2 = nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        # x = self.conv4(x, edge_index)
        # x = F.elu(x)

        # 2. Readout layer
        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin1(x)
        # x = F.elu(x)
        # x = self.lin2(x)
        return x
        # return F.log_softmax(x, dim=1)


