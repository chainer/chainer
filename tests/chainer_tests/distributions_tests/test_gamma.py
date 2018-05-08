import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


def params_init(shape):
    k = numpy.random.uniform(0, 10, shape).astype(numpy.float32)
    theta = numpy.random.uniform(0, 10, shape).astype(numpy.float32)
    params = {"k": k, "theta": theta}
    sp_params = {"a": k, "scale": theta}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.random.gamma(shape=5., size=shape).astype(numpy.float32)
    return smp

tests = set(["batch_shape", "entropy", "event_shape", "log_prob", "mean",
             "sample", "support", "variance"])


@testing.distribution_unittest(distributions.Gamma, stats.gamma,
                               params_init, sample_for_test,
                               tests=tests, support="positive")
class TestGamma(unittest.TestCase):
    pass
