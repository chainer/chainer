import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


def normal_params_init(shape):
    loc = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
    scale = numpy.exp(numpy.random.uniform(-1, 1, shape)).astype(numpy.float32)
    params = {"loc": loc, "scale": scale}
    sp_params = {"loc": loc, "scale": scale}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.random.normal(size=shape).astype(numpy.float32)
    return smp


@testing.distribution_unittest(distributions.Normal, stats.norm,
                               normal_params_init, sample_for_test)
class TestNormal(unittest.TestCase):
    pass
