import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


def params_init(shape):
    lam = numpy.random.uniform(0.1, 10, shape).astype(numpy.float32)
    params = {"lam": lam}
    sp_params = {"mu": lam}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.random.randint(0, 10, shape).astype(numpy.int32)
    return smp


tests = set(["batch_shape", "event_shape", "log_prob", "mean", "sample",
             "support", "variance"])


@testing.distribution_unittest(distributions.Poisson, stats.poisson,
                               params_init, sample_for_test,
                               tests=tests, continuous=False,
                               support="non negative integer")
class TestPoisson(unittest.TestCase):
    pass
