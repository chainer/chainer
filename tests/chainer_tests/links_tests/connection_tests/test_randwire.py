import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import links
from chainer import testing
from chainer.testing import attr


class TestRandWireWS(unittest.TestCase):

    in_channels = 3
    out_channels = 8
    _n = 16
    _k = 4
    _p = 0.75
    link = links.Convolution2D(out_channels)

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (10, self.in_channels, 5, 5)
        ).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (10, self.out_channels, 5, 5)
        ).astype(numpy.float32)
        self.l = links.RandWireWS(_n, _k, _p, link)

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = self.l(x)
        y.grad = y_grad
        y.backward()

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.l.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
