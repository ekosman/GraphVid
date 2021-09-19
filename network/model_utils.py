from network import spline_cnn
from network.SimpleGCN import SimpleGCN
from network.gat_sage import GATSage
from network.mlp_node_prediction import MLP
from network.pna import PNA
from network.relational import DynamicGCN, DynamicGCNWEdgeAttrs
from network.sage import SAGE


def get_model(num_features, num_classes, arch):
    """
    Retrieves the model corresponding to the given name.

    @param num_features: the dimensionality of the node features
    @param num_classes: number of target labels
    @param arch: name of the GNN architecture
    """
    if arch == 'sage':
        model = SAGE(num_features, 256, num_classes)
    elif arch == 'dynamic_gcn':
        model = DynamicGCNWEdgeAttrs(num_features, 256, num_classes)
    elif arch == 'gat':
        model = GATSage(num_features, num_classes)
    elif arch == 'mlp':
        model = MLP(num_features, num_classes, 2)
    elif arch == 'pna':
        model = PNA()
    elif arch == 'spline':
        model = spline_cnn.Net(num_features=num_features, num_classes=num_classes)
    elif arch == 'simple_gcn':
        model = SimpleGCN(num_node_features=num_features, num_classes=num_classes)
    else:
        raise NotImplementedError

    return model.float()
