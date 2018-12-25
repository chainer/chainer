import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


@testing.parameterize(*(testing.product_dict(
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
    testing.product({
        'dtype': [numpy.float16, numpy.float32, numpy.float64],
        'contain_zero': [True, False],
    }),
) + testing.product({
    'shape': [(0, 3)],
    'axis': [-2, 1, None],
    'dtype': [numpy.float64],
    'contain_zero': [False],
})))
class TestCumprod(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-2, 2, self.shape).astype(self.dtype)
        if self.contain_zero:
            index = numpy.random.choice(self.x.size)
            self.x.ravel()[index] = 0
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.axis is None:
            self.gy = self.gy.ravel()

        self.check_forward_options = {}
        self.check_backward_options = {}
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {
                'atol': 1e-1, 'rtol': 1e-1, 'eps': 0.01}
        elif self.dtype == numpy.float32:
            self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-3}

    def check_forward(self, x_data, axis):
        xp = backend.get_array_module(x_data)
        x = chainer.Variable(x_data)
        y = functions.cumprod(x, axis=axis)
        assert y.data.dtype == self.dtype
        y_expect = xp.asarray(numpy.cumprod(self.x, axis=axis))
        testing.assert_allclose(y_expect, y.data,
                                **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.axis)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), self.axis)

    def check_backward(self, x_data, axis, y_grad):
        gradient_check.check_backward(
            lambda x: functions.cumprod(x, axis), x_data, y_grad,
            dtype=numpy.float64, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.axis, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), self.axis, cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, axis, y_grad, x_grad_grad):
        def f(x):
            return functions.cumprod(x, axis)

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
class TestCumprodInvalidTypeAxis(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def check_type_error(self, x):
        with self.assertRaises(type_check.InvalidType):
            functions.cumprod(x, self.axis)

    def test_type_error_cpu(self):
        self.check_type_error(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        self.check_type_error(cuda.to_gpu(self.x))


class TestCumprodInvalidTypeError(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def test_invalid_type_axis(self):
        with self.assertRaises(TypeError):
            functions.cumprod(self.x, [0])
        with self.assertRaises(TypeError):
            functions.cumprod(self.x, (0,))


testing.run_module(__name__, __file__)
