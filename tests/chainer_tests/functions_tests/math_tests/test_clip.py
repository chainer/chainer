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
    'x_min_max': [
        (-0.75, 1.53),
        (numpy.float32(-0.75), numpy.float32(1.53)),
        (-1, 2),
        (None, 2),
        (-1, None),
        (None, numpy.float32(1.53)),
        (numpy.float32(-0.75), None),
    ]
}))
class TestClip(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-3, 3, self.shape).astype(self.dtype)
        # Avoid values around x_min and x_max for stability of numerical
        # gradient
        x_min, x_max = self.x_min_max
        x_min = float(x_min) if x_min is not None else self.x.min()
        x_max = float(x_max) if x_max is not None else self.x.max()
        eps = 0.01
        for ind in numpy.ndindex(self.x.shape):
            if x_min - eps < self.x[ind] < x_min + eps:
                self.x[ind] = -0.5
            elif x_max - eps < self.x[ind] < x_max + eps:
                self.x[ind] = 0.5
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

    def check_forward(self, x_data):
        x_min, x_max = self.x_min_max
        x = chainer.Variable(x_data)
        y = functions.clip(x, x_min, x_max)
        self.assertEqual(y.data.dtype, self.dtype)

        y_expect = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if (x_min is not None) and (self.x[i] < x_min):
                y_expect[i] = x_min
            elif (x_max is not None) and (self.x[i] > x_max):
                y_expect[i] = x_max

        testing.assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        def f(x):
            x_min, x_max = self.x_min_max
            return functions.clip(x, x_min, x_max)

        gradient_check.check_backward(
            f, x_data, y_grad, dtype=numpy.float64)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, gx_grad):
        def f(x):
            x_min, x_max = self.x_min_max
            return functions.clip(x, x_min, x_max)

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
        with self.assertRaises(ValueError):
            functions.clip(self.x, 1.0, -1.0)

    def test_max_min_none(self):
        with self.assertRaises(ValueError):
            functions.clip(self.x, None, None)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestClipBorderGrad(unittest.TestCase):

    def setUp(self):
        self.x = numpy.arange(6, dtype=self.dtype)
        self.x_min = 1.0
        self.x_max = 4.0
        self.expected = numpy.asarray([0, 1, 1, 1, 1, 0], dtype=self.dtype)

    def check_border_grad(self, x, expected):
        x = chainer.Variable(x)
        y = functions.clip(x, self.x_min, self.x_max)
        l = functions.sum(y)
        l.backward()
        testing.assert_allclose(x.grad, expected, atol=0, rtol=0)

    def test_border_grad_cpu(self):
        self.check_border_grad(self.x, self.expected)

    @attr.gpu
    def test_border_grad_gpu(self):
        self.check_border_grad(cuda.to_gpu(self.x), cuda.to_gpu(self.expected))


testing.run_module(__name__, __file__)
