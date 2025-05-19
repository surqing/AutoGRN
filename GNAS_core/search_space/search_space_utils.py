import torch
import dgl.nn.pytorch as dglnn
import torch.nn as nn


def conv_map(conv_type, input_dim, hidden_dim):
    if conv_type == 'SAGEConv':
        conv_layer = dglnn.SAGEConv(input_dim, hidden_dim, aggregator_type='mean')
    elif conv_type == 'SAGEv2Conv':
        conv_layer = dglnn.SAGEConv(input_dim, hidden_dim, aggregator_type='pool')
    elif conv_type == 'GCNConv':
        conv_layer = dglnn.SAGEConv(input_dim, hidden_dim, aggregator_type='gcn')
    elif conv_type == 'GATConv':
        heads = 2
        conv_layer = dglnn.GATConv(input_dim, hidden_dim, num_heads=heads, allow_zero_in_degree=True)
    elif conv_type == 'GINConv':
        apply_func = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        conv_layer = dglnn.GINConv(apply_func)
    elif conv_type == 'GraphConv':
        conv_layer = dglnn.GraphConv(input_dim, hidden_dim, allow_zero_in_degree=True)
    elif conv_type == 'GATv2Conv':
        heads = 2
        conv_layer = dglnn.GATv2Conv(input_dim, hidden_dim, num_heads=heads, allow_zero_in_degree=True)
    elif conv_type == 'TAGConv':
        conv_layer = dglnn.TAGConv(input_dim, hidden_dim)
    elif conv_type == 'EdgeConv':
        conv_layer = dglnn.EdgeConv(input_dim, hidden_dim, allow_zero_in_degree=True)
    else:
        raise Exception('Wrong convolution function!')
    return conv_layer


def bi_conv_map(bi_conv_type, input_dim, hidden_dim):
    if bi_conv_type == 'BiSAGEConv':
        bi_conv_layer = dglnn.HeteroGraphConv({
            ('gene', 'have', 'cell'): dglnn.SAGEConv(input_dim, hidden_dim * 2, aggregator_type='mean'),
            ('cell', 'have', 'gene'): dglnn.SAGEConv(input_dim, hidden_dim, aggregator_type='mean'),
        })
    elif bi_conv_type == 'BiSAGEv2Conv':
        bi_conv_layer = dglnn.HeteroGraphConv({
            ('gene', 'have', 'cell'): dglnn.SAGEConv(input_dim, hidden_dim * 2, aggregator_type='pool'),
            ('cell', 'have', 'gene'): dglnn.SAGEConv(input_dim, hidden_dim, aggregator_type='pool'),
        })
    elif bi_conv_type == 'BiGCNConv':
        bi_conv_layer = dglnn.HeteroGraphConv({
            ('gene', 'have', 'cell'): dglnn.SAGEConv(input_dim, hidden_dim * 2, aggregator_type='gcn'),
            ('cell', 'have', 'gene'): dglnn.SAGEConv(input_dim, hidden_dim, aggregator_type='gcn'),
        })
    elif bi_conv_type == 'BiGATConv':
        heads = 2
        bi_conv_layer = dglnn.HeteroGraphConv({
            ('gene', 'have', 'cell'): dglnn.GATConv(input_dim, hidden_dim * 2, num_heads=heads, allow_zero_in_degree=True),
            ('cell', 'have', 'gene'): dglnn.GATConv(input_dim, hidden_dim, num_heads=heads, allow_zero_in_degree=True),
        })
    elif bi_conv_type == 'BiGINConv':
        apply_func_gc = nn.Sequential(nn.Linear(input_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim * 2))
        apply_func_cg = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        bi_conv_layer = dglnn.HeteroGraphConv({
            ('gene', 'have', 'cell'): dglnn.GINConv(apply_func_gc),
            ('cell', 'have', 'gene'): dglnn.GINConv(apply_func_cg),
        })
    elif bi_conv_type == 'BiGraphConv':
        bi_conv_layer = dglnn.HeteroGraphConv({
            ('gene', 'have', 'cell'): dglnn.GraphConv(input_dim, hidden_dim * 2, allow_zero_in_degree=True),
            ('cell', 'have', 'gene'): dglnn.GraphConv(input_dim, hidden_dim, allow_zero_in_degree=True),
        })
    elif bi_conv_type == 'BiNoneConv':
        bi_conv_layer = None
    else:
        raise Exception('Wrong bipartite convolution function!')
    return bi_conv_layer


def act_map(act_type):
    if act_type == 'elu':
        return torch.nn.functional.elu
    elif act_type == 'tanh':
        return torch.tanh
    elif act_type == 'relu':
        return torch.nn.functional.relu
    elif act_type == 'relu6':
        return torch.nn.functional.relu6
    elif act_type == 'leaky_relu':
        return torch.nn.functional.leaky_relu
    elif act_type == 'hardtanh':
        return torch.nn.functional.hardtanh
    elif act_type == 'celu':
        return torch.nn.functional.celu
    elif act_type == 'rrelu':
        return torch.nn.functional.rrelu
    else:
        raise Exception('Wrong activate function!')


class fusion_map(nn.Module):
    def __init__(self, fusion_type, hidden_dim, out_dim):
        super().__init__()
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        hidden_dim_input = self.hidden_dim * 2 if self.fusion_type == 'concat' else self.hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim_input, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(self, src_emb, dst_emb):
        if self.fusion_type == 'concat':
            emb = torch.cat([src_emb, dst_emb], dim=1)
        elif self.fusion_type == 'multiply':
            emb = torch.multiply(src_emb, dst_emb)
        elif self.fusion_type == 'sum':
            emb = torch.add(src_emb, dst_emb)
        elif self.fusion_type == 'max':
            emb = torch.max(src_emb, dst_emb)
        elif self.fusion_type == 'abs_difference':
            emb = torch.abs(src_emb-dst_emb)
        else:
            raise Exception('Wrong fusion operation!')

        score = self.mlp(emb)
        return score
