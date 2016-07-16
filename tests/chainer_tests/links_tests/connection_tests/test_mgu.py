import unittest

import numpy

import chainer
from chainer import cuda
from chainer import links
from chainer import testing
from chainer.testing import attr


class TestMGU(unittest.TestCase):

    in_size = 4
    out_size = 5

    def setUp(self):
        self.h = numpy.random.uniform(
            -1, 1, (3, self.out_size)).astype(numpy.float32)
        self.x = numpy.random.uniform(
            -1, 1, (3, self.in_size)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (3, self.out_size)).astype(numpy.float32)

        self.mgu = links.MGU(self.in_size, self.out_size)

    def check_forward(self, h_data, x_data):
        h = chainer.Variable(h_data)
        x = chainer.Variable(x_data)
        self.mgu(h, x)

    def test_forward_cpu(self):
        self.check_forward(self.h, self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.mgu.to_gpu()
        self.check_forward(cuda.to_gpu(self.h), cuda.to_gpu(self.x))


class TestStatefulMGU(unittest.TestCase):

    in_size = 4
    out_size = 5

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (3, self.in_size)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (3, self.out_size)).astype(numpy.float32)

        self.mgu = links.StatefulMGU(self.in_size, self.out_size)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        self.mgu(x)
        self.mgu(x)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.mgu.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)
