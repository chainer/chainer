import unittest

import chainer
from chainer.backends import cuda
from chainer import distributions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
import numpy


@testing.parameterize(*testing.product({
    'shape': [(2, 3), ()],
    'dtype': [numpy.float32, numpy.float64],
}))
class TestModifiedXLogX(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(
            0.1, 1, size=self.shape).astype(self.dtype)
        self.zero_x = numpy.zeros(shape=self.shape).astype(self.dtype)
        self.gy = numpy.random.normal(size=self.shape).astype(self.dtype)
        self.ggx = numpy.random.normal(size=self.shape).astype(self.dtype)
        self.backward_options = {'atol': 1e-2, 'rtol': 1e-2, 'eps': 1e-5}

    def check_forward(self, x_data):
        distributions.utils._modified_xlogx(x_data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            distributions.utils._modified_xlogx,
            x_data, y_grad, **self.backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad):
        gradient_check.check_double_backward(
            distributions.utils._modified_xlogx, x_data, y_grad,
            x_grad_grad, dtype=numpy.float64, **self.backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
                                   cuda.to_gpu(self.ggx))

    def check_backward_zero_input(self, x_data):
        x = chainer.Variable(x_data)
        y = distributions.utils._modified_xlogx(x)
        if numpy.prod(y.shape) > 1:
            y = chainer.functions.sum(y)
        with testing.assert_warns(RuntimeWarning):
            y.backward()

    def test_backward_zero_input_cpu(self):
        self.check_backward_zero_input(self.zero_x)

    @attr.gpu
    def test_backward_zero_input_gpu(self):
        self.check_backward_zero_input(cuda.to_gpu(self.zero_x))


testing.run_module(__name__, __file__)
