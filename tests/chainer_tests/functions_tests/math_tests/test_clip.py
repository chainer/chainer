import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestClip(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # Avoid values around x_min and x_max for stability of numerical
        # gradient
        for ind in numpy.ndindex(self.x.shape):
            if -0.76 < self.x[ind] < -0.74:
                self.x[ind] = -0.5
            elif 0.74 < self.x[ind] < 0.76:
                self.x[ind] = 0.5
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.x_min = -0.75
        self.x_max = 0.75

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.clip(x, self.x_min, self.x_max)
        self.assertEqual(y.data.dtype, self.dtype)

        y_expect = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] < self.x_min:
                y_expect[i] = self.x_min
            elif self.x[i] > self.x_max:
                y_expect[i] = self.x_max

        testing.assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        def f(x):
            return functions.clip(x, self.x_min, self.x_max)

        gradient_check.check_backward(
            f, x_data, y_grad, dtype=numpy.float64)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, gx_grad):
        def f(x):
            return functions.clip(x, self.x_min, self.x_max)

        gradient_check.check_double_backward(
            f, x_data, y_grad, gx_grad, dtype=numpy.float64, atol=1e-3)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))


class TestClipInvalidInterval(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def test_invalid_interval(self):
        with self.assertRaises(AssertionError):
            functions.clip(self.x, 1.0, -1.0)


testing.run_module(__name__, __file__)
