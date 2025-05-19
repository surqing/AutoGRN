class SearchSpace(object):
    """
    Loading the search space dict
    """
    def __init__(self):

        self.stack_gnn_architecture = ['conv', 'bi_conv', 'activation', 'hidden_dim', 'fusion']

        self.space_dict = {
            'conv': ['SAGEConv', 'SAGEv2Conv', 'GCNConv', 'GATConv', 'GINConv', 'GraphConv', 'GATv2Conv', 'TAGConv', 'EdgeConv'],
            'bi_conv': ['BiSAGEConv', 'BiSAGEv2Conv', 'BiGCNConv', 'BiGATConv', 'BiGINConv', 'BiGraphConv', 'BiNoneConv'],
            'activation': ['elu', 'tanh', 'relu', 'relu6', 'leaky_relu', 'hardtanh', 'celu', 'rrelu'],
            'hidden_dim': [64, 128, 256, 512, 1024],
            'fusion': ['concat', 'multiply', 'sum', 'max', 'abs_difference']
        }
