import networkx as nx

from ge import DeepWalk


def test_DeepWalk():
    G = nx.read_edgelist('./tests/Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    print('ok1')
    model = DeepWalk(G, walk_length=3, num_walks=2, workers=1)
    print('ok2')
    model.train(window_size=3, iter=1)
    print('ok3')
    embeddings = model.get_embeddings()
    print('ok4')


if __name__ == "__main__":
    pass
