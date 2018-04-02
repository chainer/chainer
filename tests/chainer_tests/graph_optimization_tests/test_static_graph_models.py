import os
import tempfile
import unittest

import numpy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.graph_optimizations.static_graph import static_graph
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer.serializers import npz
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


class StaticMLP(chainer.Chain):

    def __init__(self, n_out):
        super(StaticMLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_out)

    @static_graph
    def __call__(self, x):
        return F.relu(self.l1(x))

class DynamicMLP(chainer.Chain):

    def __init__(self, n_out):
        super(DynamicMLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_out)

    def __call__(self, x):
        return F.relu(self.l1(x))

class TestSimpleChain(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.in_units = 5
        self.out_units = 6
        self.dynamic_chain = DynamicMLP(self.out_units)

    def test_forward(self):
        self.assertTrue(True)


testing.run_module(__name__, __file__)

#if __name__ == '__main__':
#    unittest.main()