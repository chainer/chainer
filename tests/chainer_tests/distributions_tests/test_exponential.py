import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


def params_init(shape):
    lam = numpy.exp(numpy.random.uniform(-1, 1, shape)).astype(numpy.float32)
    params = {"lam": lam}
    sp_params = {"scale": 1 / lam}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.exp(numpy.random.normal(size=shape)).astype(numpy.float32)
    return smp

tests = set(["batch_shape", "cdf", "entropy", "event_shape", "log_prob",
             "mean", "sample", "support", "variance"])


@testing.distribution_unittest(distributions.Exponential, stats.expon,
                               params_init, sample_for_test,
                               tests=tests, support="positive")
class TestExponential(unittest.TestCase):
    pass
