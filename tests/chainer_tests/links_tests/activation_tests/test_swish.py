import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _sigmoid(x):
    xp = backend.get_array_module(x)
    half = x.dtype.type(0.5)
    return xp.tanh(x * half) * half + half


class TestSwishSingle(unittest.TestCase):

    def setUp(self):
        self.x_shape = (4, 3, 2)
        self.dtype = numpy.float32

        self.link = links.Swish(())
        beta = self.link.beta.data
        beta[...] = numpy.random.uniform(-1, 1, beta.shape)
        self.link.cleargrads()

        self.x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x)
        self.assertEqual(y.data.dtype, self.dtype)

        beta_data = self.link.beta.data
        y_expect = x_data * _sigmoid(beta_data * x_data)

        testing.assert_allclose(y_expect, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, gy_data):
        gradient_check.check_backward(
            self.link, x_data, gy_data, self.link.beta, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestSwishFull(TestSwishSingle):

    def setUp(self):
        self.x_shape = (4, 3, 2)
        self.dtype = numpy.float32

        self.link = links.Swish(None)
        self.link.cleargrads()

        self.x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x)
        self.assertEqual(y.data.dtype, self.dtype)
        self.assertEqual(self.link.beta.shape, self.x_shape[1:])

        beta_data = self.link.beta.data
        y_expect = x_data * _sigmoid(beta_data * x_data)

        testing.assert_allclose(y_expect, y.data)


testing.run_module(__name__, __file__)
