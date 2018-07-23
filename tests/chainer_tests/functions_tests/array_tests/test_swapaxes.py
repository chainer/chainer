import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'in_shape': [(3, 4, 2)],
    'axis1': [0],
    'axis2': [1],
    'dtype': [numpy.float16, numpy.float32, numpy.float32],
}))
class TestSwapaxes(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(
            0.5, 1, self.in_shape).astype(self.dtype)
        self.g = numpy.random.uniform(
            0.5, 1, self.in_shape).astype(self.dtype)
        self.g = self.g.swapaxes(self.axis1, self.axis2)
        self.gg = numpy.random.uniform(
            0.5, 1, self.in_shape).astype(self.dtype)

    def check_forward(self, x_data):
        axis1, axis2 = self.axis1, self.axis2
        x = chainer.Variable(x_data)
        y = functions.swapaxes(x, axis1, axis2)
        self.assertEqual(y.data.dtype, self.dtype)
        self.assertTrue((self.x.swapaxes(axis1, axis2) ==
                         cuda.to_cpu(y.data)).all())

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.swapaxes(x, self.axis1, self.axis2)
        y.grad = y.data
        y.backward()
        testing.assert_allclose(x.data, x.grad, atol=0, rtol=0)

    def test_backward_cpu(self):
        self.check_backward(self.x)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x))

    @condition.retry(3)
    def check_double_backward(self, x_data, g_data, gg_data):
        def f(x):
            return functions.swapaxes(x, self.axis1, self.axis2)

        gradient_check.check_double_backward(
            f, x_data, g_data, gg_data, dtype=numpy.float64,
            atol=5e-2, rtol=5e-3)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.g, self.gg)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.g),
                                   cuda.to_gpu(self.gg))


testing.run_module(__name__, __file__)
