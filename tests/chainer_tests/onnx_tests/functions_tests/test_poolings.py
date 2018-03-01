import unittest

import chainer
import chainer.functions as F
from chainer import testing
import numpy as np

import onnx_chainer


@testing.parameterize(
    {'ops': F.average_pooling_2d, 'in_shape': (1, 3, 6, 6), 'args': [2, 1, 0]},
    {'ops': F.average_pooling_nd, 'in_shape': (1, 3, 6, 6, 6),
     'args': [2, 1, 0]},
    {'ops': F.max_pooling_2d, 'in_shape': (1, 3, 6, 6), 'args': [2, 1, 0]},
    {'ops': F.max_pooling_nd, 'in_shape': (1, 3, 6, 6, 6),
     'args': [2, 1, 0]},

)
class TestPoolings(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops, args):
                super(Model, self).__init__()
                self.ops = ops
                self.args = args

            def __call__(self, x):
                x = F.identity(x)
                return self.ops(*([x] + self.args))

        self.model = Model(self.ops, self.args)
        self.x = np.ones(self.in_shape, dtype=np.float32)

    def test_export_test(self):
        chainer.config.train = False
        onnx_chainer.export(self.model, self.x)

    def test_export_train(self):
        chainer.config.train = True
        onnx_chainer.export(self.model, self.x)
