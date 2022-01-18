import dgl
import torch
from dgl.nn.pytorch import GatedGraphConv,GATConv
from torch import nn


class GGNNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding_dim = args['embedding_size']
        self.gnn = GatedGraphConv(self.embedding_dim, self.embedding_dim, 8, 4)

    def forward(self, graph: dgl.DGLGraph, feats: torch.Tensor):
        return self.gnn(graph, feats, graph.edata.pop('etype'))
