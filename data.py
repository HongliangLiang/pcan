import os

import dgl
import pandas
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from c_extractor.extractor import DATA, CHILD, SUBTOKEN


class TreeDataset(Dataset):
    # PAD_WORD = 0  # special pad word id
    # UNK_WORD = 1  # out-of-vocabulary word id

    def __init__(self, args, mode):

        super(TreeDataset, self).__init__()
        tp = args['type']
        self.mode = mode
        self.args = args

        if args['dev']:
            data_path = os.path.join('data', args['dataset_name'], "dev", 'pairs.pkl')
        else:
            data_path = os.path.join('data', args['dataset_name'], 'pairs.pkl')

        print(data_path)
        assert os.path.exists(data_path)
        pairs = pandas.read_pickle(data_path)  # type: pandas.DataFrame
        train, test = train_test_split(pairs, train_size=0.6, shuffle=True, random_state=666)
        # 如果采用数据均衡策略
        if self.mode == 'train':
            if args['balanced']:
                train, test = train_test_split(pairs, train_size=0.6, shuffle=True, random_state=666)
                one_data = train[train['label'] == 1]
                zero_data = train[train['label'] == 0]
                zero_data = zero_data.sample(n=len(one_data))
                banlanced = pandas.concat([zero_data, one_data])
                pairs = banlanced.sample(frac=1)
                print(f'total train: {train.shape}--banlanced train: {pairs.shape} -- test: {test.shape}')
            else:
                pairs = train.sample(frac=1)
                print(f'total train: {train.shape}--test: {test.shape}')

        elif self.mode == 'eval':
            pairs = pandas.read_csv('data/bcb/test122283_origin.csv')
        else:
            raise

        if args['dataset_name'] in ['bcb', 'origin_bcb']:
            pairs = pairs[pairs['label'].isin([0, int(tp)])].copy()
            pairs.loc[pairs['label'] > 0, 'label'] = 1
        self.pairs = pairs
        print(f'data shape: {self.pairs.shape}')
        print(f'{self.pairs["label"].unique()}')
        self.process_tree = os.path.join('data', args['dataset_name'], 'processed.pkl')
        self.trees = pandas.read_pickle(self.process_tree)

        print('{} Dataset creation finished. #Pairs: {}'.format(self.mode, len(self.pairs)))

    def __getitem__(self, index: int):
        p = self.pairs.iloc[index]
        li, ri, label = p['id1'], p['id2'], p['label']
        lg, rg = self.trees.loc[li], self.trees.loc[ri]

        return (self.get_single(lg), self.get_single(rg)), \
               torch.tensor(label, dtype=torch.float)

    def get_single(self, gs):
        import networkx as nx
        p = nx.readwrite.node_link_graph(gs)  # type:nx.DiGraph

        if self.args['no_data']:
            r = []
            for u, v in p.edges():
                if p.edges[u, v]['etype'] == DATA:
                    r.append((u, v))
            p.remove_edges_from(r)

        # if self.args['no_pos']:
        #     r = []
        #     for u, v in p.edges():
        #         if p.edges[u, v]['etype'] == CHILD:
        #             r.append((u, v))
        #     p.remove_edges_from(r)

        dg = dgl.from_networkx(p, node_attrs=['x', 'mask'], edge_attrs=['etype'])

        return dg

    def __len__(self) -> int:
        return len(self.pairs)


class D4C(Dataset):
    def __init__(self, args, mode):

        super(D4C, self).__init__()
        # tp = args['type']
        self.mode = mode
        self.args = args

        if args['dev']:
            data_path = os.path.join('data', args['dataset_name'], "dev4c", 'processed4c.pkl')
        else:
            data_path = os.path.join('data', args['dataset_name'], 'processed4c.pkl')

        trees = pandas.read_pickle(data_path)

        print(data_path)

        train, test = train_test_split(trees, train_size=0.6, shuffle=True, random_state=666)
        if self.mode == 'train':
            d = train
        elif self.mode == 'eval':
            d, _ = train_test_split(test, train_size=0.5, shuffle=True, random_state=666)
        elif self.mode == 'test':
            _, d = train_test_split(test, train_size=0.5, shuffle=True, random_state=666)
        else:
            raise

        self.trees = d

        print('{} Dataset creation finished. #Pairs: {}'.format(self.mode, len(self.trees)))

    def get_single(self, gs):
        import networkx as nx
        p = nx.readwrite.node_link_graph(gs)  # type:nx.DiGraph

        if self.args['no_data']:
            r = []
            for u, v in p.edges():
                if p.edges[u, v]['etype'] == DATA:
                    r.append((u, v))
            p.remove_edges_from(r)

        # if self.args['no_pos']:
        #     r = []
        #     for u, v in p.edges():
        #         if p.edges[u, v]['etype'] == CHILD:
        #             r.append((u, v))
        #     p.remove_edges_from(r)

        dg = dgl.from_networkx(p, node_attrs=['x', 'mask'], edge_attrs=['etype'])

        return dg

    def __getitem__(self, index: int):
        p = self.trees.iloc[index]

        return self.get_single(p['graph']), torch.tensor(p['label'] - 1, dtype=torch.long)

    def __len__(self):
        return len(self.trees)


class BigCloneBenchClone(Dataset):
    # PAD_WORD = 0  # special pad word id
    # UNK_WORD = 1  # out-of-vocabulary word id

    def __init__(self, args, mode):

        super(BigCloneBenchClone, self).__init__()
        tp = args['type']
        self.mode = mode
        self.args = args

        if args['dev']:
 #           data_path = os.path.join('data', args['dataset_name'], "dev", 'pairs.pkl')

            data_path = os.path.join('data', args['dataset_name'], 'pairs.pkl')
            print(f'data_path: {data_path}')
        else:
            data_path = os.path.join('data', args['dataset_name'], 'pairs.pkl')

        print(data_path)
        assert os.path.exists(data_path)
        pairs = pandas.read_pickle(data_path)  # type: pandas.DataFrame

        train = pairs.loc[pairs['split'] == mode]

        if mode == 'train':
            tp = pairs.loc[pairs['label'] == 1]
            fp = pairs.loc[pairs['label'] == 0]  # type:pandas.DataFrame
            sfp = fp.sample(n=len(tp))
            balanced = pandas.concat([tp, sfp])
            self.pairs = balanced.sample(frac=1)
        else:
            self.pairs = p

        self.process_tree = os.path.join('data', args['dataset_name'], 'processed.pkl')
        self.trees = pandas.read_pickle(self.process_tree)

        print('{} Dataset creation finished. #Pairs: {}'.format(self.mode, len(self.pairs)))

    def __getitem__(self, index: int):
        p = self.pairs.iloc[index]
        li, ri, label = p['id1'], p['id2'], p['label']
        lg, rg = self.trees.loc[li], self.trees.loc[ri]

        return (self.get_single(lg), self.get_single(rg)), \
               torch.tensor(label, dtype=torch.float)

    def get_single(self, gs):
        import networkx as nx
        p = nx.readwrite.node_link_graph(gs)  # type:nx.DiGraph

        if self.args['no_data']:
            r = []
            for u, v in p.edges():
                if p.edges[u, v]['etype'] == DATA:
                    r.append((u, v))
            p.remove_edges_from(r)

        if self.args['no_child']:
            r = []
            for u, v in p.edges():
                if p.edges[u, v]['etype'] == CHILD:
                    r.append((u, v))
            p.remove_edges_from(r)
# lsp
#        if  self.args['no_subtoken']:
        if 1:
            r = []
            for u, v in p.edges():
                if p.edges[u, v]['etype'] == SUBTOKEN:
                    r.append(v)
            p.remove_nodes_from(r)

        dg = dgl.from_networkx(p, node_attrs=['x', 'mask'], edge_attrs=['etype'])

        return dg

    def __len__(self) -> int:
        return len(self.pairs)
