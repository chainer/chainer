import chainer
from chainer import function_node
from chainer import functions
from chainer import link
from chainer import variable

from itertools import product
import numpy as np
import six


class WeightedSum(function_node.FunctionNode):
    """Weighted sum of given links."""
    def forward(self, inputs):
        self.retain_inputs(six.moves.range(len(inputs)))
        w = inputs[0]
        xs = inputs[1:]
        y = sum(wi * x for wi, x in zip(w, xs))
        return y,

    def backward(self, indexes, grad_outputs):
        inputs = self.get_retained_inputs()
        w = inputs[0]
        xs = inputs[1:]
        gy, = grad_outputs
        gxs = tuple(wi * gy for wi in w)
        gws = functions.stack([functions.sum(x * gy) for x in xs])
        return (gws,) + gxs


def weighted_sum(xs, w):
    y, = WeightedSum().apply((w, *xs))
    return y


def WattsStrogatz(n, k, p):
    """A random graph generator based on Watts-Strogatz.

    Args:
        n (int): # of nodes in a random graph.
        k (int): # of connected nodes of each nodes in initial. `k` should be an
            even integer.
        p (int): The probability of rewiring.
    """
    # Generate initial edges
    edges = [None] * n
    for v in six.moves.range(n):
        cw = set((v + 1 + i) % n for i in six.moves.range(k // 2))
        ccw = set((v - 1 - i) % n for i in six.moves.range(k // 2))
        edges[v] = cw | ccw
    # Rewiring by Watts-Strogatz algorithm
    should_rewire = np.random.rand(k // 2, n) < p
    for i, v in product(six.moves.range(k // 2), six.moves.range(n)):
        if not should_rewire[i, v]:
            continue
        i = (v + 1 + i) % n
        edges[v].remove(i)
        edges[i].remove(v)
        possible_nodes = list(set(six.moves.range(n)) - edges[v] - set([v]))
        to = np.random.choice(possible_nodes)
        edges[v].add(to)
        edges[to].add(v)
    return edges


def DAG(graph):
    """A converter from undirected-graph to directed-graph. This function adds
        extra input/output nodes and wire from original input/output nodes.
    """
    N = len(graph)
    parents = [None] * (N + 2)
    parents[0] = []
    outputs = set(six.moves.range(1, N + 1))
    for v in six.moves.range(N):
        parents[v + 1] = [i + 1 for i in sorted(graph[v]) if i < v]
        if len(parents[v + 1]) == 0:
            parents[v + 1] = [0]
        outputs = outputs - set(parents[v + 1])
    parents[-1] = sorted(outputs)
    return parents


class RandWire(link.ChainList):
    """A random wired neural network module.

    See: `Exploring Randomly Wired Neural Networks for Image Recognition\
          <https://arxiv.org/abs/1904.01569>`_

    Args:
        DAG (list of list): A DAG denotes wired neural network. Each list in DAG
            has IDs of parents of the node. DAG[0] denotes extra-input node and
            DAG[-1] denotes extra-output node.
        link (~chainer.Link): A link module in each node.
        aggregator (function or ~chainer.Function): A function module used to
            aggregate multiple nodes. If `weighted == True`, aggregator will be
            ignored.
        weighted (bool): Weighted sum or not, in aggregation.
    """
    def __init__(self, DAG, link, aggregator=sum, weighted=True,
                 *args, **kwargs):
        super(RandWire, self).__init__()
        self.aggregator = aggregator
        self.weighted = weighted
        self.DAG = DAG
        self.num = len(DAG)
        self.nodes = [None] * self.num
        for v in six.moves.range(1, self.num - 1):
            self.nodes[v] = link(*args, **kwargs)
            self.append(self.nodes[v])
        self.nodes[-1] = lambda x: x
        if self.weighted:
            with self.init_scope():
                n_link = sum(len(vs) for vs in self.DAG)
                self.weights = variable.Parameter(0, n_link)

    def __call__(self, x):
        if self.weighted:
            ws = functions.sigmoid(self.weights)
            j = 0
        hs = [None] * self.num
        hs[0] = x
        for i in six.moves.range(1, self.num):
            xs = [hs[k] for k in self.DAG[i]]
            if self.weighted:
                h = weighted_sum(xs, ws[j:j + len(xs)])
                j += len(xs)
            else:
                h = self.aggregator(xs)
            hs[i] = self.nodes[i](h)
        return hs[-1]


def RandWireWS(n, k, p, link, aggregator=sum, weighted=True, *args, **kwargs):
    """RandWire based on Watts-Strogatz."""
    graph = DAG(WattsStrogatz(n, k, p))
    return RandWire(graph, link, aggregator, weighted, *args, **kwargs)
