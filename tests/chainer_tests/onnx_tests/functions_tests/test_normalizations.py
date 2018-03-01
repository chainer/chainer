import unittest

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing
import numpy as np

import onnx_chainer


@testing.parameterize(
    {'ops': F.local_response_normalization,
     'input_argname': 'x',
     'args': {'n': 5, 'k': 2, 'alpha': 1e-4, 'beta': 0.75}},
)
class TestNormalizations(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops, args, input_argname):
                super(Model, self).__init__()
                self.ops = ops
                self.args = args
                self.input_argname = input_argname

            def __call__(self, x):
                x = F.identity(x)
                self.args[self.input_argname] = x
                return self.ops(**self.args)

        self.model = Model(self.ops, self.args, self.input_argname)
        self.x = np.zeros((1, 5), dtype=np.float32)

    def test_export_test(self):
        chainer.config.train = False
        onnx_chainer.export(self.model, self.x)

    def test_export_train(self):
        chainer.config.train = True
        onnx_chainer.export(self.model, self.x)


class TestBatchNormalization(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.bn = L.BatchNormalization(5)

            def __call__(self, x):
                x = F.identity(x)
                return self.bn(x)

        self.model = Model()
        self.x = np.zeros((1, 5), dtype=np.float32)

    def test_export_test(self):
        chainer.config.train = False
        onnx_chainer.export(self.model, self.x)

    def test_export_train(self):
        chainer.config.train = True
        onnx_chainer.export(self.model, self.x)
