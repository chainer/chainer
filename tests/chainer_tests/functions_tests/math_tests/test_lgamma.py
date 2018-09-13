import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
import chainer.functions as F
from chainer import testing


def _lgamma_cpu(x, dtype):
    from scipy import special
    return numpy.vectorize(special.gammaln, otypes=[dtype])(x)


def _lgamma_gpu(x, dtype):
    return cuda.to_gpu(_lgamma_cpu(cuda.to_cpu(x), dtype))


def _lgamma_expected(x, dtype):
    if backend.get_array_module(x) is numpy:
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
@testing.with_requires('scipy')
class TestLGamma(unittest.TestCase):
    pass


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
}))
@testing.without_requires('scipy')
class TestLGammaExceptions(unittest.TestCase):
    def setUp(self):
        self.x, self.gy, self.ggx = make_data(self.shape, self.dtype)
        self.func = F.lgamma

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        with self.assertRaises(ImportError):
            self.func(x)

    def test_forward_cpu(self):
        self.check_forward(self.x)


testing.run_module(__name__, __file__)
