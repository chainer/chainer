import unittest

import numpy

import chainer
from chainer import cuda
from chainer import links
from chainer import testing
from chainer.testing import attr


class TestSGU(unittest.TestCase):

    in_size = 4
    out_size = 8

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (3, self.in_size)).astype(numpy.float32)
        self.h = numpy.random.uniform(
            -1, 1, (3, self.out_size)).astype(numpy.float32)
        self.sgu = links.SGU(self.in_size, self.out_size)

    def check_forward(self, h_data, x_data):
        h = chainer.Variable(h_data)
        x = chainer.Variable(x_data)

        self.sgu(h, x)

    def test_forward_cpu(self):
        self.check_forward(self.h, self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.sgu.to_gpu()
        self.check_forward(cuda.to_gpu(self.h), cuda.to_gpu(self.x))


class TestDSGU(unittest.TestCase):

    in_size = 4
    out_size = 8

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (3, self.in_size)).astype(numpy.float32)
        self.h = numpy.random.uniform(
            -1, 1, (3, self.out_size)).astype(numpy.float32)
        self.sgu = links.DSGU(self.in_size, self.out_size)

    def check_forward(self, h_data, x_data):
        h = chainer.Variable(h_data)
        x = chainer.Variable(x_data)

        self.sgu(h, x)

    def test_forward_cpu(self):
        self.check_forward(self.h, self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.sgu.to_gpu()
        self.check_forward(cuda.to_gpu(self.h), cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)
