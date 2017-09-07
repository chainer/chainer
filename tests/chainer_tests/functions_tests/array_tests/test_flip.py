import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (1,), 'axis': 0},
        {'shape': (2, 3, 4), 'axis': 0},
        {'shape': (2, 3, 4), 'axis': 1},
        {'shape': (2, 3, 4), 'axis': 2},
        {'shape': (2, 3, 4), 'axis': -3},
        {'shape': (2, 3, 4), 'axis': -2},
        {'shape': (2, 3, 4), 'axis': -1},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
class TestFlip(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(0, 1, self.shape).astype(self.dtype)
        self.g = numpy.random.uniform(0, 1, self.shape).astype(self.dtype)

    def check_forward(self, x_data, axis):
        x = chainer.Variable(x_data)
        y = functions.flip(x, axis)

        testing.assert_allclose(y.data, numpy.flip(x_data, axis))

    def test_forward_cpu(self):
        self.check_forward(self.x, self.axis)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), self.axis)

    def check_backward(self, x_data, axis, y_grad):
        gradient_check.check_backward(
            functions.Flip(axis), x_data, y_grad, dtype=numpy.float64)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.axis, self.g)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), self.axis, cuda.to_gpu(self.g))


@testing.parameterize(
    {'axis': 3},
    {'axis': -4},
)
class TestFlipInvalidTypeAxis(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def check_type_error(self, x):
        with self.assertRaises(type_check.InvalidType):
            functions.flip(x, self.axis)

    def test_type_error_cpu(self):
        self.check_type_error(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        self.check_type_error(cuda.to_gpu(self.x))


class TestFlipInvalidTypeError(unittest.TestCase):

    def test_invalid_axis(self):
        with self.assertRaises(TypeError):
            functions.Flip('a')


testing.run_module(__name__, __file__)
