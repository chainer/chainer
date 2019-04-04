import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
import chainermn.functions


@testing.parameterize(*testing.product({
    'shape_x': [[(4, 5), (3, 2)], [(3, 2)], [()]],
    'shape_delegate': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestPseudoConnect(unittest.TestCase):

    def setUp(self):
        self.delegate = numpy.random.uniform(-1, 1, self.shape_delegate)\
            .astype(self.dtype)
        self.x = tuple([
            numpy.random.uniform(-1, 1, shape).astype(self.dtype)
            for shape in self.shape_x])
        self.gy = tuple([
            numpy.random.uniform(-1, 1, shape).astype(self.dtype)
            for shape in self.shape_x])

    def check_forward(self, delegate_data, x_data):
        delegate_variable = chainer.Variable(delegate_data)
        x = tuple([chainer.Variable(data) for data in x_data])

        y = chainermn.functions.pseudo_connect(delegate_variable, *x)
        for _y in y:
            self.assertEqual(_y.data.dtype, self.dtype)
        for _x, _y in zip(self.x, y):
            y_expect = _x.copy()
            testing.assert_allclose(y_expect, _y.data)

    def test_forward_cpu(self):
        self.check_forward(self.delegate, self.x)

    @attr.gpu
    def test_forward_gpu(self):
        x = tuple([cuda.to_gpu(_x) for _x in self.x])
        self.check_forward(cuda.to_gpu(self.delegate), x)

    def check_backward(self, delegate_data, x_data, y_grad):
        gradient_check.check_backward(
            chainermn.functions.pseudo_connect,
            (delegate_data, ) + x_data, y_grad,
            dtype=numpy.float64)

    def test_backward_cpu(self):
        self.check_backward(self.delegate, self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        x = tuple([cuda.to_gpu(_x) for _x in self.x])
        gy = tuple([cuda.to_gpu(_gy) for _gy in self.gy])
        self.check_backward(cuda.to_gpu(self.delegate), x, gy)
