import unittest

from chainer import testing
import chainer.functions as F
import numpy


def make_data(shape, dtype):
    x = numpy.random.uniform(-0.9, 0.9, shape).astype(dtype, copy=False)
    gy = numpy.random.uniform(-1, 1, shape).astype(dtype, copy=False)
    ggx = numpy.random.uniform(-1, 1, shape).astype(dtype, copy=False)
    return x, gy, ggx


@testing.unary_math_function_unittest(F.arctanh, make_data=make_data)
class TestArctanh(unittest.TestCase):
    pass
