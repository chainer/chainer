import unittest

import numpy

from chainer import function_node
from chainer import testing
from chainer.utils import precision


class F(function_node.FunctionNode):

    @precision._fp16_mixed_precision_helper
    def forward(self, x):
        self.x = x
        return x


class G(function_node.FunctionNode):

    @precision._fp16_mixed_precision_helper
    def forward(self, x):
        return None,


class TestMixedPrecision(unittest.TestCase):

    def test_fp16(self):
        x = (numpy.zeros((1, 2, 3), dtype=numpy.float16),
             numpy.zeros((1, 2, 3), dtype=numpy.float16))
        f = F()
        y = f.apply(x)
        assert f.x[0].dtype == numpy.float32
        assert f.x[1].dtype == numpy.float32
        assert y[0].dtype == numpy.float16
        assert y[1].dtype == numpy.float16

    def test_fp32(self):
        x = (numpy.zeros((1, 2, 3), dtype=numpy.float32),
             numpy.zeros((1, 2, 3), dtype=numpy.float32))
        f = F()
        y = f.apply(x)
        assert f.x[0] is x[0]
        assert f.x[1] is x[1]
        assert y[0].dtype == numpy.float32
        assert y[1].dtype == numpy.float32

    def test_fp64(self):
        x = (numpy.zeros((1, 2, 3), dtype=numpy.float64),
             numpy.zeros((1, 2, 3), dtype=numpy.float64))
        f = F()
        y = f.apply(x)
        assert f.x[0] is x[0]
        assert f.x[1] is x[1]
        assert y[0].dtype == numpy.float64
        assert y[1].dtype == numpy.float64

    def test_float16_int8(self):
        x = (numpy.zeros((1, 2, 3), dtype=numpy.float16),
             numpy.zeros((1, 2, 3), dtype=numpy.int8))
        f = F()
        y = f.apply(x)
        assert f.x[0].dtype == numpy.float32
        assert f.x[1] is x[1]
        assert y[0].dtype == numpy.float16
        assert y[1].dtype == numpy.int8

    def test_none(self):
        x = numpy.zeros((1, 2, 3), dtype=numpy.float64),
        y = G().apply(x)
        assert y[0].data is None


testing.run_module(__name__, __file__)
