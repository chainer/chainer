import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import testing
from chainer.testing import attr


class MockDistribution(object):

    def __init__(self, x):
        self.x = x
        self.h_call_count = 0
        self.y_call_count = 0

    @chainer.distribution.cached_property
    def h(self):
        self.h_call_count += 1
        return self.x * 2

    @chainer.distribution.cached_property
    def y(self):
        self.y_call_count += 1
        return self.h * 3


class TestCachedProperty(unittest.TestCase):

    def test1(self):
        obj = MockDistribution(chainer.Variable(numpy.array([1.])))
        obj.h
        obj.h
        obj.h
        assert obj.h_call_count == 1

    def test2(self):
        obj = MockDistribution(chainer.Variable(numpy.array([1.])))
        with chainer.no_backprop_mode():
            obj.h
            obj.h
            obj.h
        assert obj.h_call_count == 1


testing.run_module(__name__, __file__)
