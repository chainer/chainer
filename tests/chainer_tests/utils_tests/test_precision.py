import unittest

import numpy

from chainer import function_node
from chainer import testing
from chainer import utils


class F(function_node.FunctionNode):

    @utils.mixed_precision
    def forward(self, x):
        self.x = x
        return x


class TestMixedPrecision(unittest.TestCase):

    def test_fp16(self):
        x = (numpy.zeros((1, 2, 3), dtype=numpy.float16),) * 2
        f = F()
        y = f.apply(x)
        assert f.x[0].dtype == numpy.float32
        assert f.x[1].dtype == numpy.float32
        assert y[0].dtype == numpy.float16
        assert y[1].dtype == numpy.float16

    def test_fp32(self):
        x = (numpy.zeros((1, 2, 3), dtype=numpy.float32),) * 2
        f = F()
        y = f.apply(x)
        assert f.x[0] is x[0]
        assert f.x[1] is x[1]
        assert y[0].dtype == numpy.float32
        assert y[1].dtype == numpy.float32

    def test_fp64(self):
        x = (numpy.zeros((1, 2, 3), dtype=numpy.float64),) * 2
        f = F()
        y = f.apply(x)
        assert f.x[0] is x[0]
        assert f.x[1] is x[1]
        assert y[0].dtype == numpy.float64
        assert y[1].dtype == numpy.float64

    def test_int8(self):
        x = (numpy.zeros((1, 2, 3), dtype=numpy.int8),) * 2
        with self.assertRaises(AssertionError):
            F().apply(x)


testing.run_module(__name__, __file__)
