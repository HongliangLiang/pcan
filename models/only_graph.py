import dgl
import torch
import torchtext
from dgl.nn.pytorch import GatedGraphConv, GlobalAttentionPooling
from torch import nn


class AttentionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, contexts, mask):
        """
        :param contexts: (batch_sz, input_length, context_sz)
        :param mask: (batch_sz, input_length)
        :return: attn shape: (batch_sz, input_length, 1)
        """

        mask = mask.float().masked_fill(mask, float('-inf')).unsqueeze(-1)
        attention = self.attention(contexts) + mask
        attention = torch.softmax(attention, dim=1)
        code_vector = torch.sum(contexts * attention, dim=1)
        return code_vector


class TreeTransformer(nn.Module):

    def __init__(self, args, vocab: torchtext.vocab.Vocab = None, embedding=None):
        super(TreeTransformer, self).__init__()
        if vocab is not None:
            padding_idx = vocab.stoi['<pad>']
            self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False,
                                                          padding_idx=padding_idx)  # type:nn.Embedding
        elif embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False, padding_idx=1)
        else:
            exit(1)
        self.in_feats = args['in_feats']
        self.out_feats = args['out_feats']
        self.n_steps = args['n_steps']
        self.ggnn = args['ggnn']
        self.n_etypes = args['n_etypes']
        self.gap = args['gap']

        self.word_dropout = nn.Dropout(0.2)

        self.max_statement = getattr(args, 'ms', 80)

        assert self.in_feats == self.embedding.embedding_dim
        self.tree_encoder = GatedGraphConv(self.in_feats, self.out_feats, n_steps=self.n_steps,
                                           n_etypes=self.n_etypes)

        self.attention = AttentionLayer(d_model=self.out_feats)

        self.gate_fn = torch.nn.Linear(self.out_feats, 1)
        self.output_layer = GlobalAttentionPooling(gate_nn=self.gate_fn)

        self.device = torch.device(args['device'])

    def forward(self, graph: dgl.DGLGraph):
        feats = self.embedding(graph.ndata.pop('x'))

        feats = self.word_dropout(feats)

        etypes = graph.edata.pop('etype')

        graph.ndata['h'] = self.tree_encoder(graph, feats, etypes)

        return self.output_layer(graph, graph.ndata.pop('h'))


# from models.avg_max import TreeTransformer


class CodeCloneClassifier(nn.Module):

    def __init__(self, args, vocab=None, embedding=None):
        super().__init__()

        # in_feats = args['embedding_size']
        out_feats = args['out_feats']

        self.tree_transformer = TreeTransformer(args, vocab, embedding)

        self.device = torch.device(args['device'])

        self.fc = torch.nn.Sequential(
            nn.BatchNorm1d(out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats // 2),
            nn.BatchNorm1d(out_feats // 2),
            nn.ReLU(),
            nn.Linear(out_feats // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        right: dgl.DGLGraph
        left: dgl.DGLGraph
        left, right = x

        left = left.to(self.device)
        right = right.to(self.device)

        l = self.tree_transformer(left)
        r = self.tree_transformer(right)
        abs_dist = torch.abs(torch.sub(l, r))

        return self.fc(abs_dist).squeeze(-1)
