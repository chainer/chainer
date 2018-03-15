import unittest

import numpy

import chainer.functions as F
from chainer import testing


# sqrt

def make_data(shape, dtype):
    x = numpy.random.uniform(0.1, 5, shape).astype(dtype)
    gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
    ggx = numpy.random.uniform(-1, 1, shape).astype(dtype)
    return x, gy, ggx


@testing.unary_math_function_unittest(
    F.sqrt,
    make_data=make_data,
    backward_options={'eps': 1e-3},
    double_backward_options={'eps': 1e-3},
)
class TestSqrt(unittest.TestCase):
    pass


# rsqrt

def rsqrt(x, dtype):
    return numpy.reciprocal(numpy.sqrt(x, dtype=dtype), dtype=dtype)


@testing.unary_math_function_unittest(
    F.rsqrt,
    func_expected=rsqrt,
    make_data=make_data,
    forward_options={'atol': 1e-2},
    backward_options={'eps': 1e-2, 'atol': 1e-2, 'rtol': 1e-2},
    double_backward_options={'eps': 1e-2, 'atol': 1e-1, 'rtol': 1e-1},
)
class TestRsqrt(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
