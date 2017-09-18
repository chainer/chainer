import random
import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestSELU(unittest.TestCase):

    def setUp(self):
        # Avoid unstability of numeraical grad
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.x[(-0.01 < self.x) & (self.x < 0.01)] = 0.5
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.alpha = random.random()
        self.scale = random.random()
        self.check_forward_options = {}
        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.selu(x, alpha=self.alpha, scale=self.scale)
        self.assertEqual(y.data.dtype, self.dtype)

        expected = numpy.where(
            self.x >= 0, self.x, self.alpha * (numpy.exp(self.x) - 1))
        expected *= self.scale

        testing.assert_allclose(
            expected, y.data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            lambda x: functions.selu(x, alpha=self.alpha, scale=self.scale),
            x_data, y_grad, dtype=numpy.float64,
            **self.check_backward_options)

    @condition.retry(10)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(10)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
