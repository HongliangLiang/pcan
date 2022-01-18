from typing import List

import dgl
import torch
import torchtext
from onmt.utils.misc import sequence_mask
from torch import nn
from onmt.encoders import RNNEncoder

from models.transformer import TransformerEncoder


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

        self.max_statement = getattr(args, 'ms', 256)

        assert self.in_feats == self.embedding.embedding_dim

        self.attention = AttentionLayer(d_model=self.out_feats)

        num_layers = args['num_layers']
        nhead = args['nhead']
        self.output_layer = TransformerEncoder(num_layers=num_layers, d_model=self.out_feats,
                                               heads=nhead, d_ff=2048, dropout=0.1,
                                               attention_dropout=0,
                                               max_relative_positions=80)

        self.device = torch.device(args['device'])

    def pad_indexes(self, indexes, clipped_lengths, max_length):
        return torch.tensor([idx[:length] + [0] * (max_length - length)
                             for idx, length in zip(indexes, clipped_lengths)],
                            device=self.device)

    def forward(self, graph: dgl.DGLGraph):
        feats = self.embedding(graph.ndata.pop('x'))

        graph.ndata['h'] = self.word_dropout(feats)

        graphs: List[dgl.DGLGraph] = dgl.unbatch(graph)

        real_lengths: List[int] = []
        indexes = []

        for g in graphs:
            mask: torch.Tensor = g.ndata['mask']
            idx = mask.nonzero(as_tuple=True)[0].tolist()
            indexes.append(idx)
            real_lengths.append(len(idx))

        max_length = min(max(real_lengths), self.max_statement)

        clipped_lengths = [min(max_length, length) for length in real_lengths]

        padded_indexes = self.pad_indexes(indexes, clipped_lengths, max_length)

        statement_vectors = torch.stack(
            [torch.index_select(graph.ndata['h'], 0, idx)
             for graph, idx in zip(graphs, padded_indexes)])

        lens = torch.tensor(clipped_lengths, device=self.device)

        out = self.output_layer(statement_vectors, lens)  # shape: batch_size X seq_len X model_dim

        code_vector = self.attention(out, ~sequence_mask(lens, max_length))  # shape: (batch_sz, seq_length, 1)

        return code_vector


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
