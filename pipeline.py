import os
import pickle
from collections import Counter

import networkx as nx
import pandas as pd
from gensim.models import Word2Vec
from networkx.readwrite import json_graph
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vectors, Vocab

from c_extractor.extractor import DATA, SUBTOKEN, PARENT, CHILD
from c_extractor.extractor import GeneralVisitor
from utils import get_params

pandarallel.initialize()


def parse_graph(ast):
    v = GeneralVisitor()
    v.visit(ast)
    x = v.json()
    return x


def get_nx(p: pd.Series):
    nodes = p['nodes']
    mask = p['mask']
    edges = p['edges']
    etype = p['etype']
    g = nx.DiGraph()
    for k, (x, m) in enumerate(zip(nodes, mask)):
        g.add_node(k, x=x, mask=m)

    for e, t in zip(edges, etype):
        if t in [DATA, CHILD]:
            g.add_edge(*e, etype=t)
        elif t in [SUBTOKEN, PARENT]:
            g.add_edge(*reversed(e), etype=t)
        else:
            raise ValueError("unknown edge type{}".format(t))
    return json_graph.node_link_data(g)


class Pipeline:
    """
    train word2vec word embedding
    return torchtext vocab
    get statement trees and its path to root
    """

    def __init__(self, args):

        self.args = args
        self.embedding_size = args['embedding_size']
        self.vocab_size = args['vocab_size']

        self.data_dir = "data"
        #self.data_dir = '/home/hdj/lsp_temp/pcan/data/tbccd_3'
        assert os.path.exists(self.data_dir)
        self.copra_path = os.path.join(self.data_dir, args['dataset_name'], "raw.pkl")
        print('copra path:{}'.format(self.copra_path))

        if not os.path.exists(self.copra_path):

            if args['dataset_name'] in ['ojclone']:

                source_path = os.path.join(self.data_dir, args['dataset_name'], 'programs.pkl')
                source = pd.read_pickle(source_path)
                from pycparser import c_parser
                parser = c_parser.CParser()
                parsed_file = os.path.join(self.data_dir, args['dataset_name'], 'parsed.pkl')
                if not os.path.exists(parsed_file):
                    print('parsing...')
                    source.columns = ['id', 'src', 'label']

                    source['tree'] = source['src'].apply(parser.parse)
                    source.to_pickle(parsed_file)
                else:
                    print('already parsed...')
                    source = pd.read_pickle(parsed_file)

                graph = source['tree'].apply(parse_graph)

                graph.columns = ['nodes', 'mask', 'edges', 'etype']
                graph['label'] = source['label']

                graph[['nodes', 'mask', 'edges', 'etype', 'label']].to_pickle(self.copra_path)

            elif args['dataset_name'] in ['bcb', 'tbccd', 'tbccd_2', 'tbccd_3', 'fa-ast']:
                bcb = os.path.join(self.data_dir, args['dataset_name'], "bcb.tsv")
                raw = pd.read_csv(bcb, sep='\t', names=['id', 'nodes', 'mask', 'edges', 'etype'])  # type: pd.DataFrame
                raw['id'] = raw['id'].astype(int)
                raw.set_index('id', inplace=True)
                import json
                raw['nodes'] = raw['nodes'].parallel_apply(str.split)
                raw['mask'] = raw['mask'].parallel_apply(json.loads)
                raw['edges'] = raw['edges'].parallel_apply(json.loads)
                raw['etype'] = raw['etype'].parallel_apply(json.loads)
                raw.to_pickle(self.copra_path)

        self.dict_path = os.path.join(self.data_dir, args['dataset_name'], "embedding",
                                      '{}.dict'.format(args['dataset_name']))
        self.vector_path = os.path.join(self.data_dir, args['dataset_name'], "embedding",
                                        '{}.vec'.format(args['dataset_name']))
        self.data_path = os.path.join(self.data_dir, args['dataset_name'], 'processed.pkl')  # final data

        if not os.path.exists(self.data_path) or not os.path.exists(self.vector_path):
            self.train_word2vec_vectors()
            self.vocab = self.get_torchtext_vocab()
            self.tree2index()

    def tree2index(self):
        # translate path to idx

        def nodes2idx(nodes):
            return [self.vocab.stoi[n] for n in nodes]

        print('start to get statement trees and root paths ...')
        data = pd.read_pickle(self.copra_path)
        data['nodes'] = data['nodes'].apply(nodes2idx)

        data.to_pickle(self.data_path)

        if self.args['dataset_name'] in ['bcb', 'origin_bcb']:
            d = data.apply(get_nx, axis=1)
            d.to_pickle(self.data_path)
        else:
            data = data.apply(get_nx, axis=1)
            data.to_pickle(self.data_path)

        print('save data to {}'.format(self.data_path))

    def prepare_sentences(self):

        print('preparing sentence...')
        p = pd.read_pickle(self.copra_path)  # type: pd.DataFrame
        if self.args['task'] == 'classification':
            trees, _ = train_test_split(p, train_size=0.6, shuffle=True, random_state=666)
        elif self.args['task'] == 'clone':
            if self.args['dataset_name'] not in ['tbccd', 'tbccd_2', 'tbccd_3', 'fa-ast']:
                data_path = os.path.join('data', self.args['dataset_name'], 'pairs.pkl')
                pairs = pd.read_pickle(data_path)
                pairs, _ = train_test_split(pairs, train_size=0.6, shuffle=True, random_state=666)
                train_ids = pairs['id1'].append(pairs['id2']).unique()
                trees = p.loc[p.index.intersection(train_ids)]
            else:
                data_path = os.path.join('data', self.args['dataset_name'], 'pairs.pkl')
                pairs = pd.read_pickle(data_path)
                pairs = pairs.loc[pairs['split'] == 'train']
                train_ids = pairs['id1'].append(pairs['id2']).unique()
                trees = p.loc[p.index.intersection(train_ids)]
        else:
            raise

        copra = trees['nodes']
        return copra

    def get_torchtext_vocab(self) -> Vocab:
        vec = Vectors(name=self.vector_path)
        with open(self.dict_path, 'rb') as f:
            word2count = pickle.load(f)
        vocab: Vocab = Vocab(Counter(word2count), vectors=vec)
        return vocab

    def train_word2vec_vectors(self):
        sentences = self.prepare_sentences()
        print('total sentences: {}'.format(len(sentences)))
        print('training word2vec embedding...')
        w2v = Word2Vec(sentences, sg=1, size=self.embedding_size, workers=16,
                       max_final_vocab=self.vocab_size)
        dir = os.path.dirname(self.vector_path)
        if not os.path.exists(dir):
            os.mkdir(dir)
        abs_path = os.path.abspath(self.vector_path)
        w2v.wv.save_word2vec_format(abs_path)
        print('save word vectors to {}'.format(self.vector_path))
        # get word frequency dict
        w2c = dict()
        for item in w2v.wv.vocab:
            w2c[item] = w2v.wv.vocab[item].count

        with open(self.dict_path, 'wb') as f:
            pickle.dump(w2c, f)
        print('save dict to {}, total words: {}'.format(self.dict_path, len(w2c)))


if __name__ == '__main__':
    args2 = vars(get_params())

    p = Pipeline(args2)
