import unittest

import numpy

import chainer
from chainer import testing
from chainer.utils import cache


class MockDistribution(object):

    def __init__(self, x):
        self.x = x
        self.h_call_count = 0
        self.y_call_count = 0

    @cache.cached_property
    def h(self):
        self.h_call_count += 1
        return self.x * 2

    @cache.cached_property
    def y(self):
        self.y_call_count += 1
        return self.h * 3


class TestCachedProperty(unittest.TestCase):

    def test_name(self):
        assert MockDistribution.y.__name__ == 'y'

    def test1(self):
        obj = MockDistribution(chainer.Variable(numpy.array([1.])))
        h0 = obj.h
        h1 = obj.h
        assert obj.h_call_count == 1
        assert h0 is h1
        numpy.testing.assert_allclose(h0.array, 2.)

    def test2(self):
        obj = MockDistribution(chainer.Variable(numpy.array([1.])))
        with chainer.no_backprop_mode():
            h0 = obj.h
            h1 = obj.h
        assert obj.h_call_count == 1
        assert h0 is h1
        numpy.testing.assert_allclose(h0.array, 2.)

    def test3(self):
        obj = MockDistribution(chainer.Variable(numpy.array([1.])))
        h0 = obj.h
        with chainer.no_backprop_mode():
            h1 = obj.h
        h2 = obj.h
        with chainer.no_backprop_mode():
            h3 = obj.h
        assert obj.h_call_count <= 2
        assert h0 is h2
        assert h0 is not h1
        assert h1 is h3
        numpy.testing.assert_allclose(h0.array, 2.)
        numpy.testing.assert_allclose(h1.array, 2.)

    def test_attrs1(self):
        obj = MockDistribution(chainer.Variable(numpy.array([1.])))
        h0 = obj.h
        y0 = obj.y
        h1 = obj.h
        y1 = obj.y
        assert obj.h_call_count == 1
        assert obj.y_call_count == 1
        assert h0 is h1
        assert y0 is y1
        numpy.testing.assert_allclose(h0.array, 2.)
        numpy.testing.assert_allclose(y0.array, 6.)

    def test_objs1(self):
        obj0 = MockDistribution(chainer.Variable(numpy.array([1.])))
        obj1 = MockDistribution(chainer.Variable(numpy.array([10.])))
        y00 = obj0.y
        y10 = obj1.y
        y01 = obj0.y
        y11 = obj1.y
        assert obj0.y_call_count == 1
        assert obj1.y_call_count == 1
        assert y00 is y01
        assert y10 is y11
        numpy.testing.assert_allclose(y00.array, 6.)
        numpy.testing.assert_allclose(y10.array, 60.)


testing.run_module(__name__, __file__)
