import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
import chainer.functions as F
from chainer import testing


def _ndtri_cpu(x, dtype):
    from scipy import special
    return numpy.vectorize(special.ndtri, otypes=[dtype])(x)


def _ndtri_gpu(x, dtype):
    return cuda.to_gpu(_ndtri_cpu(cuda.to_cpu(x), dtype))


def _ndtri_expected(x, dtype):
    if backend.get_array_module(x) is numpy:
        return _ndtri_cpu(x, dtype)
    else:
        return _ndtri_gpu(x, dtype)


def make_data(shape, dtype):
    x = numpy.random.uniform(0.1, 0.9, shape).astype(dtype)
    gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
    ggx = numpy.random.uniform(-1, 1, shape).astype(dtype)
    return x, gy, ggx


@testing.unary_math_function_unittest(
    F.ndtri,
    func_expected=_ndtri_expected,
    make_data=make_data,
    forward_options={'atol': 1e-3, 'rtol': 1e-3},
    backward_options={'eps': 1e-6},
    double_backward_options={'eps': 1e-6}
)
@testing.with_requires('scipy')
class TestNdtri(unittest.TestCase):
    pass


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
}))
@testing.without_requires('scipy')
class TestNdtriExceptions(unittest.TestCase):
    def setUp(self):
        self.x, self.gy, self.ggx = make_data(self.shape, self.dtype)
        self.func = F.ndtri

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        with self.assertRaises(ImportError):
            self.func(x)

    def test_forward_cpu(self):
        self.check_forward(self.x)


testing.run_module(__name__, __file__)
