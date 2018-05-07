import unittest

import numpy
from scipy import special

from chainer.backends import cuda
import chainer.functions as F
from chainer import testing


def _lgamma_cpu(x, dtype):
    return numpy.vectorize(special.gammaln, otypes=[dtype])(x)


def _lgamma_gpu(x, dtype):
    return cuda.to_gpu(_lgamma_cpu(cuda.to_cpu(x), dtype))


def _lgamma_expected(x, dtype):
    if cuda.get_array_module(x) is numpy:
        return _lgamma_cpu(x, dtype)
    else:
        return _lgamma_gpu(x, dtype)


def make_data(shape, dtype):
    x = numpy.random.uniform(1., 10., shape).astype(dtype)
    gy = numpy.random.uniform(-1., 1., shape).astype(dtype)
    ggx = numpy.random.uniform(-1., 1., shape).astype(dtype)
    return x, gy, ggx


@testing.unary_math_function_unittest(
    F.lgamma,
    func_expected=_lgamma_expected,
    make_data=make_data,
    backward_options={'eps': 1e-3, 'atol': 5e-2, 'rtol': 1e-4,
                      'dtype': numpy.float64},
    double_backward_options={'eps': 1e-3, 'atol': 5e-2, 'rtol': 1e-4,
                             'dtype': numpy.float64}
)
class TestLGamma(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
