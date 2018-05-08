import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


def params_init(shape):
    a = numpy.random.uniform(0, 10, shape).astype(numpy.float32)
    b = numpy.random.uniform(0, 10, shape).astype(numpy.float32)
    params = {"a": a, "b": b}
    sp_params = {"a": a, "b": b}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.random.uniform(size=shape).astype(numpy.float32)
    return smp

tests = set(["batch_shape", "entropy", "event_shape", "log_prob", "mean",
             "sample", "support", "variance"])


@testing.distribution_unittest(distributions.Beta, stats.beta,
                               params_init, sample_for_test,
                               tests=tests, support="[0, 1]")
class TestBeta(unittest.TestCase):
    pass
