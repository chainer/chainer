import unittest

import numpy

from chainer.backends import cuda
import chainer.functions as F
from chainer import testing


def _log_ndtr_cpu(x, dtype):
    from scipy import special
    return special.log_ndtr(x).astype(dtype)


def _log_ndtr_gpu(x, dtype):
    return cuda.to_gpu(_log_ndtr_cpu(cuda.to_cpu(x), dtype))


def _log_ndtr_expected(x, dtype):
    if cuda.get_array_module(x) is numpy:
        return _log_ndtr_cpu(x, dtype)
    else:
        return _log_ndtr_gpu(x, dtype)


@testing.unary_math_function_unittest(
    F.log_ndtr,
    func_expected=_log_ndtr_expected,
)
@testing.with_requires('scipy')
class TestLogNdtr(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
