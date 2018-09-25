import math
import unittest

import numpy

from chainer import backend
from chainer.backends import cuda
import chainer.functions as F
from chainer import testing


def _erf_cpu(x, dtype):
    return numpy.vectorize(math.erf, otypes=[dtype])(x)


def _erf_gpu(x, dtype):
    return cuda.to_gpu(_erf_cpu(cuda.to_cpu(x), dtype))


def _erf_expected(x, dtype):
    if backend.get_array_module(x) is numpy:
        return _erf_cpu(x, dtype)
    else:
        return _erf_gpu(x, dtype)


@testing.unary_math_function_unittest(
    F.erf,
    func_expected=_erf_expected,
)
class TestErf(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
