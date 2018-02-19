import unittest

import numpy

from chainer import cuda
from chainer import functions
from chainer import gradient_check


class TestSRU(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 4)).astype('f')
        self.c = numpy.random.uniform(-1, 1, (3, 4)).astype('f')
        self.u = numpy.random.uniform(-1, 1, (3, 12)).astype('f')
        self.gc = numpy.random.uniform(-1, 1, (3, 4)).astype('f')
        self.gh = numpy.random.uniform(-1, 1, (3, 4)).astype('f')

    def check_backward(self, x, c, u, gc, gh):
        gradient_check.check_backward(
            functions.sru, (x, c, u), (gc, gh))

    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.c), cuda.to_gpu(self.u),
            cuda.to_gpu(self.gc), cuda.to_gpu(self.gh))
