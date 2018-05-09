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
    smp = numpy.random.randint(1, 10, shape).astype(numpy.int32)
    return smp

tests = set(["batch_shape", "event_shape", "log_prob", "mean", "sample",
             "support", "variance"])


@testing.distribution_unittest(distributions.Geometric, stats.geom,
                               params_init, sample_for_test,
                               tests=tests, continuous=False,
                               support="positive integer")
class TestGeometric(unittest.TestCase):
    pass
