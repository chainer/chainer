import unittest

import chainer
import numpy as np
import onnx_chainer

from chainer import testing


@testing.parameterize(
    {'info': 'Add', 'ops': 'a + b'},
    {'info': 'Sub', 'ops': 'a - b'},
    {'info': 'Mul', 'ops': 'a * b'},
    {'info': 'Neg', 'ops': '-a'},
    {'info': 'Absolute', 'ops': 'abs(a)'},
    {'info': 'Div', 'ops': 'a / b'},
    {'info': 'Clip', 'ops': 'chainer.functions.clip(a, 0.1, 0.2)'},
    {'info': 'Exp', 'ops': 'chainer.functions.exp(a)'},
    {'info': 'MatMul', 'ops': 'chainer.functions.matmul(a, b.T)'},
    {'info': 'Maximum', 'ops': 'chainer.functions.maximum(a, b)'},
    {'info': 'Minimum', 'ops': 'chainer.functions.minimum(a, b)'},
    {'info': 'Sqrt', 'ops': 'chainer.functions.sqrt(a)'},
    {'info': 'SquaredDifference',
     'ops': 'chainer.functions.squared_difference(a, b)'},
    {'info': 'PowVarConst',
     'ops': 'chainer.functions.math.basic_math.pow(a, 2)'},
    {'info': 'Sum',
     'ops': 'chainer.functions.sum(a, axis=1)'},
    {'info': 'Sum',
     'ops': 'chainer.functions.sum(a, axis=0, keepdims=True)'},
)
class TestMathOperators(unittest.TestCase):

    def setUp(self):
        class Model(chainer.Chain):

            def __init__(self, ops):
                super(Model, self).__init__()
                self.ops = ops

            def __call__(self, a, b):
                return eval(self.ops)

        self.model = Model(self.ops)
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((2, 3), dtype=np.float32) * 2
        self.x = (a, b)

    def test_export_test(self):
        chainer.config.train = False
        onnx_chainer.export(self.model, self.x)

    def test_export_train(self):
        chainer.config.train = True
        onnx_chainer.export(self.model, self.x)
