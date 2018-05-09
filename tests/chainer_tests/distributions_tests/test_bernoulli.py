import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


def params_init(shape):
    p = numpy.random.uniform(0, 1, shape).astype(numpy.float32)
    params = {"p": p}
    sp_params = {"p": p}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.random.randint(2, size=shape).astype(numpy.float32)
    return smp


tests = set(["batch_shape", "entropy", "log_prob", "mean", "prob", "sample",
             "stddev", "support", "variance"])


@testing.distribution_unittest(distributions.Bernoulli, stats.bernoulli,
                               params_init, sample_for_test,
                               tests=tests, continuous=False, support='{0, 1}')
class TestBernoulli(unittest.TestCase):
    pass
