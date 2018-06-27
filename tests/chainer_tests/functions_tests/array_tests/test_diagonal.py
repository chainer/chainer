import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (2, 4, 6), 'args': (1, 2, 0)},
        {'shape': (2, 4, 6), 'args': (-1, 2, 0)},
        {'shape': (2, 4, 6), 'args': (0, -1, -2)},
        {'shape': (2, 4, 6), 'args': (0, -1, 1)},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
class TestDiagonal(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.y_expected = self.x.diagonal(*self.args)
        self.y_shape = self.y_expected.shape
        self.gy = numpy.random.uniform(-1, 1, self.y_shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.diagonal(x, *self.args)
        testing.assert_allclose(y.data, self.y_expected)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            lambda x: functions.diagonal(x, *self.args),
            x_data, y_grad, dtype=numpy.float64)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad):
        def f(x):
            return functions.diagonal(x, *self.args)

        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad,
            atol=1e-3, rtol=1e-2, dtype=numpy.float64)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))


testing.run_module(__name__, __file__)
