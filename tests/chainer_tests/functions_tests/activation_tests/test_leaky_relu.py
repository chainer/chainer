import random
import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@testing.fix_random()
@backend.inject_backend_tests(
    ['test_forward_cpu', 'test_backward_cpu', 'test_double_backward_cpu'],
    # CPU tests
    testing.product({
        'use_ideep': ['never', 'always'],
    }))
class TestLeakyReLU(unittest.TestCase):

    def setUp(self):
        # Avoid unstability of numeraical grad
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.x[(-0.05 < self.x) & (self.x < 0.05)] = 0.5
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.slope = random.random()
        self.check_forward_options = {}
        self.check_backward_options = {}
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-2}

    def check_forward(self, x_data, backend_config):
        x = chainer.Variable(x_data)
        with backend_config:
            y = functions.leaky_relu(x, slope=self.slope)
        self.assertEqual(y.data.dtype, self.dtype)

        expected = numpy.where(self.x >= 0, self.x, self.x * self.slope)

        testing.assert_allclose(
            expected, y.data, **self.check_forward_options)

    def test_forward_cpu(self, backend_config):
        self.check_forward(self.x, backend_config)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad, backend_config):
        def f(x):
            with backend_config:
                return functions.leaky_relu(x, self.slope)

        gradient_check.check_backward(
            f, x_data, y_grad, dtype=numpy.float64,
            **self.check_backward_options)

    def test_backward_cpu(self, backend_config):
        self.check_backward(self.x, self.gy, backend_config)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad,
                              x_grad_grad, backend_config):
        def f(x):
            with backend_config:
                y = functions.leaky_relu(x, self.slope)
                return y * y
        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad, dtype=numpy.float64,
            **self.check_double_backward_options)

    def test_double_backward_cpu(self, backend_config):
        self.check_double_backward(self.x, self.gy, self.ggx, backend_config)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x),
            cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx))


testing.run_module(__name__, __file__)
