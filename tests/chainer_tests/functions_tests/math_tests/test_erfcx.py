import unittest

import numpy

from chainer import backend
from chainer.backends import cuda
import chainer.functions as F
from chainer import testing


def _erfcx_cpu(x, dtype):
    from scipy import special
    return special.erfcx(x).astype(dtype)


def _erfcx_gpu(x, dtype):
    return cuda.to_gpu(_erfcx_cpu(cuda.to_cpu(x), dtype))


def _erfcx_expected(x, dtype):
    if backend.get_array_module(x) is numpy:
        return _erfcx_cpu(x, dtype)
    else:
        return _erfcx_gpu(x, dtype)


@testing.unary_math_function_unittest(
    F.erfcx,
    func_expected=_erfcx_expected,
)
@testing.with_requires('scipy')
class TestErfcx(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
