import unittest

from chainer import distributions
from chainer import testing
import numpy


def params_init(shape):
    loc = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
    scale = numpy.exp(numpy.random.uniform(-1, 1, shape)).astype(numpy.float32)
    params = {"loc": loc, "scale": scale}
    sp_params = {"loc": loc, "scale": scale}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.random.normal(size=shape).astype(numpy.float32)
    return smp


tests = set(["batch_shape", "cdf", "entropy", "event_shape", "icdf",
             "log_cdf", "log_prob", "log_survival", "mean", "prob",
             "sample", "stddev", "support", "survival", "variance"])


@testing.distribution_unittest(distributions.Normal, 'norm',
                               params_init, sample_for_test,
                               tests=tests)
class TestNormal(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
