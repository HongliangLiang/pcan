import random

import dgl
import torch
from dgl.udf import NodeBatch
from torch import nn


class TreeEncoder(nn.Module):
    def __init__(self, in_feats, out_feats, n_etypes):
        super().__init__()
        self._n_etypes = n_etypes
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.w_node = nn.Linear(out_feats, out_feats, bias=True)
        self.edge_embedding = nn.Embedding(n_etypes, out_feats)
        self.act = nn.Tanh()

    def apply_node_func(self, nodes: NodeBatch):
        return {'h': self.act(self.w_node(nodes.data['h']) + nodes.data['agg'])}

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor, etypes):
        graph = graph.local_var()
        zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
        feat = torch.cat([feat, zero_pad], -1)
        graph.ndata['h'] = feat
        graph.edata['e'] = self.edge_embedding(etypes)
        graph.ndata['agg'] = feat.new_zeros((graph.number_of_nodes(), self._out_feats))

        dgl.prop_nodes_topo(graph=graph,
                            message_func=dgl.function.u_mul_e('h', 'e', 'm'),
                            reduce_func=dgl.function.sum('m', 'agg'),
                            apply_node_func=self.apply_node_func)
        return graph.ndata['h']


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    seed = 26

    set_seed(seed)

    t = TreeEncoder(2, 2, 2)
    g = dgl.DGLGraph()
    g.add_nodes(5)
    g.add_edges([0, 1, 3, 4], [1, 2, 2, 2])
    feats = torch.tensor([[0, 0], [1, 1, ], [2, 2, ], [3, 3], [4, 4]]).float()
    etypes = torch.tensor([0, 0, 1, 1])

    x = t(g, feats, etypes)
    print('feats', feats)
    print(x)
