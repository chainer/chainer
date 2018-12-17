import math
import unittest

import numpy

from chainer import backend
from chainer.backends import cuda
import chainer.functions as F
from chainer import testing


def _erfc_cpu(x, dtype):
    return numpy.vectorize(math.erfc, otypes=[dtype])(x)


def _erfc_gpu(x, dtype):
    return cuda.to_gpu(_erfc_cpu(cuda.to_cpu(x), dtype))


def _erfc_expected(x, dtype):
    if backend.get_array_module(x) is numpy:
        return _erfc_cpu(x, dtype)
    else:
        return _erfc_gpu(x, dtype)


@testing.unary_math_function_unittest(
    F.erfc,
    func_expected=_erfc_expected,
)
class TestErfc(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
