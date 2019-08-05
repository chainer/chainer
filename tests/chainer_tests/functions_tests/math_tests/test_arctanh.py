import unittest

import numpy

import chainer.functions as F
from chainer import testing


def make_data(shape, dtype):
    # Input values close to -1 or 1 would make tests unstable
    x = numpy.random.uniform(-0.9, 0.9, shape).astype(dtype, copy=False)
    gy = numpy.random.uniform(-1, 1, shape).astype(dtype, copy=False)
    ggx = numpy.random.uniform(-1, 1, shape).astype(dtype, copy=False)
    return x, gy, ggx


@testing.unary_math_function_unittest(
    F.arctanh,
    make_data=make_data,
    backward_options={'eps': 1e-3},
    double_backward_options={'eps': 1e-3},
)
class TestArctanh(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
