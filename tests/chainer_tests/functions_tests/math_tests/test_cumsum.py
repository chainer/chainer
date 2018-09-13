import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
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
        {'shape': (2, 3, 4), 'axis': None},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
class TestCumsum(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.axis is None:
            self.gy = self.gy.ravel()

        self.check_forward_options = {}
        self.check_backward_options = {}
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2}
            self.check_backward_options = {'atol': 1e-2}
            self.check_double_backward_options = {'atol': 1e-1, 'eps': 0.01}
        elif self.dtype == numpy.float32:
            self.check_double_backward_options = {'atol': 1e-3}

    def check_forward(self, x_data, axis):
        xp = backend.get_array_module(x_data)
        x = chainer.Variable(x_data)
        y = functions.cumsum(x, axis=axis)
        self.assertEqual(y.data.dtype, self.dtype)
        y_expect = xp.asarray(numpy.cumsum(self.x, axis=axis))
        testing.assert_allclose(y_expect, y.data,
                                **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.axis)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), self.axis)

    def check_backward(self, x_data, axis, y_grad):
        gradient_check.check_backward(
            lambda x: functions.cumsum(x, axis), x_data, y_grad,
            dtype=numpy.float64, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.axis, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), self.axis, cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, axis, y_grad, x_grad_grad):
        def f(x):
            return functions.cumsum(x, axis)

        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad, dtype=numpy.float64,
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.axis, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x), self.axis,
                                   cuda.to_gpu(self.gy),
                                   cuda.to_gpu(self.ggx))


@testing.parameterize(
    {'axis': 3},
    {'axis': -4},
)
class TestCumsumInvalidTypeAxis(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def check_type_error(self, x):
        with self.assertRaises(type_check.InvalidType):
            functions.cumsum(x, self.axis)

    def test_type_error_cpu(self):
        self.check_type_error(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        self.check_type_error(cuda.to_gpu(self.x))


class TestCumsumInvalidTypeError(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def test_invalid_type_axis(self):
        with self.assertRaises(TypeError):
            functions.cumsum(self.x, [0])
        with self.assertRaises(TypeError):
            functions.cumsum(self.x, (0,))


testing.run_module(__name__, __file__)
