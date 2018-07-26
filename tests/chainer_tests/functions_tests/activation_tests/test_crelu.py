import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


def _replace_near_zero_values(x):
    # Replace near zero values in an array in order to avoid unstability of
    # numerical grad
    x[(-0.01 < x) & (x < 0.01)] = 0.5


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (5, 4), 'y_shape': (10, 4), 'axis': 0},
        {'shape': (5, 4), 'y_shape': (5, 8), 'axis': 1},
        {'shape': (5, 4), 'y_shape': (5, 8), 'axis': -1},
        {'shape': (5, 4), 'y_shape': (10, 4), 'axis': -2},
        {'shape': (5, 4, 3, 2), 'y_shape': (10, 4, 3, 2), 'axis': 0},
        {'shape': (5, 4, 3, 2), 'y_shape': (5, 8, 3, 2), 'axis': 1},
        {'shape': (5, 4, 3, 2), 'y_shape': (5, 4, 6, 2), 'axis': 2},
        {'shape': (5, 4, 3, 2), 'y_shape': (5, 4, 3, 4), 'axis': 3},
        {'shape': (5, 4, 3, 2), 'y_shape': (5, 4, 3, 4), 'axis': -1},
        {'shape': (5, 4, 3, 2), 'y_shape': (5, 4, 6, 2), 'axis': -2},
        {'shape': (5, 4, 3, 2), 'y_shape': (5, 8, 3, 2), 'axis': -3},
        {'shape': (5, 4, 3, 2), 'y_shape': (10, 4, 3, 2), 'axis': -4},
    ], [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
@testing.fix_random()
class TestCReLU(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        _replace_near_zero_values(self.x)
        self.gy = numpy.random.uniform(
            -1, 1, self.y_shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(
            -1, 1, self.shape).astype(self.dtype)
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_double_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.crelu(x, axis=self.axis)
        self.assertEqual(y.data.dtype, self.dtype)
        self.assertEqual(y.data.shape, self.y_shape)

        expected_former = numpy.maximum(self.x, 0)
        expected_latter = numpy.maximum(-self.x, 0)
        expected = numpy.concatenate(
            (expected_former, expected_latter), axis=self.axis)
        testing.assert_allclose(expected, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        def f(x):
            return chainer.functions.crelu(x, self.axis)

        gradient_check.check_backward(
            f, x_data, y_grad, dtype=numpy.float64)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad):
        def f(x):
            return chainer.functions.crelu(x, self.axis)

        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad, dtype=numpy.float64,
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
                                   cuda.to_gpu(self.ggx))


testing.run_module(__name__, __file__)
