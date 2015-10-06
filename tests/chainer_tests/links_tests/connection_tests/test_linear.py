import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


class TestLinear(unittest.TestCase):

    in_shape = (3,)
    out_size = 2

    def setUp(self):
        in_size = numpy.prod(self.in_shape)
        self.func = links.Linear(in_size, self.out_size)
        W = self.func.params['W'].data
        b = self.func.params['b'].data
        W[...] = numpy.random.uniform(-1, 1, W.shape).astype(numpy.float32)
        b[...] = numpy.random.uniform(-1, 1, b.shape).astype(numpy.float32)
        self.func.zerograds()

        self.W = W.copy()  # fixed on CPU
        self.b = b.copy()  # fixed on CPU

        x_shape = (4,) + self.in_shape
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (4, self.out_size)).astype(numpy.float32)
        self.y = self.x.reshape(4, -1).dot(W.T) + b

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.func(x)
        self.assertEqual(y.data.dtype, numpy.float32)
        gradient_check.assert_allclose(self.y, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.func.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        W = self.func.params['W']
        b = self.func.params['b']

        y = self.func(x)
        y.grad = y_grad
        y.backward()

        f = lambda: self.func(x)
        gx, gW, gb = gradient_check.numerical_grad(
            f, (x.data, W.data, b.data), (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(gx, x.grad)
        gradient_check.assert_allclose(gW, W.grad)
        gradient_check.assert_allclose(gb, b.grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.func.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestLinearWithSpatialDimensions(TestLinear):

    in_shape = (3, 2, 2)


class TestInvalidLinear(unittest.TestCase):

    def setUp(self):
        self.func = links.Linear(3, 2)
        self.x = numpy.random.uniform(-1, 1, (4, 1, 2)).astype(numpy.float32)

    def test_invalid_size(self):
        with self.assertRaises(type_check.InvalidType):
            self.func(chainer.Variable(self.x))
