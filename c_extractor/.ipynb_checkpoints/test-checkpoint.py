import pandas

from c_extractor.extractor import GeneralVisitor


def func(ast):
    v = GeneralVisitor()
    v.visit(ast)
    x = v.json()
    return x


if __name__ == "__main__":
    source = pandas.read_pickle('./dev.pkl')
    source.columns = ['id', 'src', 'label', 'tree']
    graph = source['tree'].apply(func)
    graph.columns = ['nodes', 'mask', 'edges', 'etype']
    graph['label'] = source['label']

    graph[['nodes', 'mask', 'edges', 'etype', 'label']].to_pickle('graph.pkl')
