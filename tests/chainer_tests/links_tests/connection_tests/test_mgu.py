import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import links
from chainer import testing
from chainer.testing import attr


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def mgu(W_f, W_h, h, x):
    f = sigmoid(numpy.concatenate([h, x]).dot(W_f.T))
    hx = numpy.concatenate([f * h, x])
    h_bar = numpy.tanh(hx.dot(W_h.T))
    h_new = f * h_bar + (1 - f) * h
    return h_new


class TestStatelessMGU(unittest.TestCase):

    in_size = 4
    out_size = 5

    def setUp(self):
        self.h = numpy.random.uniform(
            -1, 1, (3, self.out_size)).astype(numpy.float32)
        self.x = numpy.random.uniform(
            -1, 1, (3, self.in_size)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (3, self.out_size)).astype(numpy.float32)

        self.mgu = links.StatelessMGU(self.in_size, self.out_size)

    def check_forward(self, h_data, x_data):
        h = chainer.Variable(h_data)
        x = chainer.Variable(x_data)
        y = self.mgu(h, x)

        W_f = cuda.to_cpu(self.mgu.W_f.W.data)
        W_h = cuda.to_cpu(self.mgu.W_h.W.data)
        for i in six.moves.range(3):
            h_new = mgu(W_f, W_h, self.h[i], self.x[i])
            testing.assert_allclose(h_new, y.data[i])

    def test_forward_cpu(self):
        self.check_forward(self.h, self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.mgu.to_gpu()
        self.check_forward(cuda.to_gpu(self.h), cuda.to_gpu(self.x))


@testing.parameterize(
    {'dtype': numpy.float16,
     'forward_tols': {'atol': 5e-4, 'rtol': 5e-3}},
    {'dtype': numpy.float32,
     'forward_tols': {'atol': 1e-5, 'rtol': 1e-4}},
    {'dtype': numpy.float64,
     'forward_tols': {'atol': 1e-5, 'rtol': 1e-4}},
)
class TestStatefulMGU(unittest.TestCase):

    in_size = 4
    out_size = 5

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (3, self.in_size)).astype(self.dtype)
        self.gy = numpy.random.uniform(
            -1, 1, (3, self.out_size)).astype(self.dtype)

        with chainer.using_config('dtype', self.dtype):
            self.mgu = links.StatefulMGU(self.in_size, self.out_size)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        W_f = cuda.to_cpu(self.mgu.W_f.W.data)
        W_h = cuda.to_cpu(self.mgu.W_h.W.data)
        with chainer.using_config('dtype', self.dtype):
            y1 = self.mgu(x)
            y2 = self.mgu(x)

        h = numpy.zeros(self.out_size, dtype=self.dtype)
        for i in six.moves.range(3):
            h1 = mgu(W_f, W_h, h, self.x[i])
            testing.assert_allclose(h1, y1.data[i], **self.forward_tols)
            h2 = mgu(W_f, W_h, h1, self.x[i])
            testing.assert_allclose(h2, y2.data[i], **self.forward_tols)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.mgu.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)
