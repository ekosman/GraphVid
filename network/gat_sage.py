import torch
from torch_geometric.nn import GATConv
from tqdm import tqdm
import torch.nn.functional as F


class GATSage(torch.nn.Module):
    def __init__(self, num_features, num_classes, n_heads=2):
        super(GATSage, self).__init__()
        self.n_heads = n_heads
        self.num_layers = 2
        self.conv1 = GATConv(num_features, 8, heads=self.n_heads)
        self.conv2 = GATConv(8*self.n_heads, num_classes, heads=1)
        self.convs = torch.nn.ModuleList([self.conv1, self.conv2])

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all