import unittest

import numpy as np
import onnx

import chainer
from chainer import testing
from onnx_chainer.testing import test_onnxruntime


@testing.parameterize(
    {'info': 'Neg', 'ops': '-a'},
    {'info': 'Absolute', 'ops': 'abs(a)'},
    {'info': 'Clip', 'ops': 'chainer.functions.clip(a, 0.1, 0.2)'},
    {'info': 'Exp', 'ops': 'chainer.functions.exp(a)'},
    {'info': 'Sqrt', 'ops': 'chainer.functions.sqrt(a)'},
    {'info': 'PowVarConst',
     'ops': 'chainer.functions.math.basic_math.pow(a, 2)'},
    {'info': 'Sum',
     'ops': 'chainer.functions.sum(a, axis=1)'},
    {'info': 'Sum',
     'ops': 'chainer.functions.sum(a, axis=0, keepdims=True)'},
    {'info': 'AddConst', 'ops': 'a + 1'},
    {'info': 'Max', 'ops': 'chainer.functions.max(a, axis=0)'},
    {'info': 'Mean', 'ops': 'chainer.functions.mean(a, axis=0)'},
    {'info': 'Min', 'ops': 'chainer.functions.min(a, axis=0)'},
    {'info': 'Prod', 'ops': 'chainer.functions.prod(a, axis=0)'},
    {'info': 'LogSumExp', 'ops': 'chainer.functions.logsumexp(a, axis=0)'},
)
class TestUnaryMathOperators(unittest.TestCase):

    def setUp(self):
        class Model(chainer.Chain):

            def __init__(self, ops):
                super(Model, self).__init__()
                self.ops = ops

            def __call__(self, a):
                if not isinstance(a, chainer.Variable):
                    a = chainer.Varaible(a)
                return eval(self.ops)

        self.model = Model(self.ops)
        self.a = chainer.Variable(np.ones((2, 3), dtype=np.float32))
        self.fn = self.info + '.onnx'

    def test_output(self):
        for opset_version in range(
                test_onnxruntime.MINIMUM_OPSET_VERSION,
                onnx.defs.onnx_opset_version() + 1):
            test_onnxruntime.check_output(
                self.model, self.a, self.fn, opset_version=opset_version)


@testing.parameterize(
    {'info': 'Add', 'ops': 'a + b'},
    {'info': 'Sub', 'ops': 'a - b'},
    {'info': 'Mul', 'ops': 'a * b'},
    {'info': 'Div', 'ops': 'a / b'},
    {'info': 'MatMul', 'ops': 'chainer.functions.matmul(a, b, transb=True)'},
    {'info': 'Maximum', 'ops': 'chainer.functions.maximum(a, b)'},
    {'info': 'Minimum', 'ops': 'chainer.functions.minimum(a, b)'},
)
class TestBinaryMathOperators(unittest.TestCase):

    def setUp(self):
        class Model(chainer.Chain):

            def __init__(self, ops):
                super(Model, self).__init__()
                self.ops = ops

            def __call__(self, a, b):
                if not isinstance(a, chainer.Variable):
                    a = chainer.Varaible(a)
                if not isinstance(b, chainer.Variable):
                    b = chainer.Varaible(b)
                return eval(self.ops)

        self.model = Model(self.ops)
        a = chainer.Variable(np.ones((2, 3), dtype=np.float32))
        b = chainer.Variable(np.ones((2, 3), dtype=np.float32) * 2)
        self.x = (a, b)
        self.fn = self.info + '.onnx'

    def test_output(self):
        for opset_version in range(
                test_onnxruntime.MINIMUM_OPSET_VERSION,
                onnx.defs.onnx_opset_version() + 1):
            test_onnxruntime.check_output(
                self.model, self.x, self.fn, opset_version=opset_version)


@testing.parameterize(
    {'info': 'LinearInterpolate',
     'ops': 'chainer.functions.linear_interpolate(a, b, c)'},
)
class TestTernaryMathOperators(unittest.TestCase):

    def setUp(self):
        class Model(chainer.Chain):

            def __init__(self, ops):
                super(Model, self).__init__()
                self.ops = ops

            def __call__(self, a, b, c):
                if not isinstance(a, chainer.Variable):
                    a = chainer.Varaible(a)
                if not isinstance(b, chainer.Variable):
                    b = chainer.Varaible(b)
                if not isinstance(c, chainer.Variable):
                    c = chainer.Varaible(c)
                return eval(self.ops)

        self.model = Model(self.ops)
        a = chainer.Variable(np.ones((2, 3), dtype=np.float32))
        b = chainer.Variable(np.ones((2, 3), dtype=np.float32) * 2)
        c = chainer.Variable(np.ones((2, 3), dtype=np.float32) * 3)
        self.x = (a, b, c)
        self.fn = self.info + '.onnx'

    def test_output(self):
        for opset_version in range(
                test_onnxruntime.MINIMUM_OPSET_VERSION,
                onnx.defs.onnx_opset_version() + 1):
            test_onnxruntime.check_output(
                self.model, self.x, self.fn, opset_version=opset_version)
