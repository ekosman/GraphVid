import torch
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    A module which consists of several n_steps fully connected layers with ReLU activations
    between every 2 layers. The ouput of the module has no activation
    """

    def __init__(self, in_features, out_features, n_steps, dropout=None, bias=True):
        super(MLP, self).__init__()
        steps = np.linspace(in_features, out_features, n_steps + 1, dtype=int)
        steps[0] = in_features
        steps[-1] = out_features
        # steps = np.linspace(in_features, out_out_features, n_steps + 1, dtype=int)
        layers = OrderedDict()
        for idx, (in_f, out_f) in enumerate(zip(steps[: -1], steps[1:])):
            layers[f"linear_{idx}"] = nn.Linear(in_features=in_f, out_features=out_f, bias=bias)
            layers[f"relu_{idx}"] = nn.ReLU()
            if dropout is not None:
                layers[f"dropout_{idx}"] = nn.Dropout(dropout)

        if dropout is not None:
            del layers[list(layers.keys())[-1]]
        del layers[list(layers.keys())[-1]]
        self.net = nn.Sequential(layers)

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        (_, _, size) = adjs[-1]
        x_target = x[:size[1]]  # Target nodes are always placed first.
        x = self.net(x_target)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        return self.net(x_all)
