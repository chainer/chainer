import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


def params_init(shape):
    low = numpy.random.uniform(-10, 0, shape).astype(numpy.float32)
    high = numpy.random.uniform(low, low + 10, shape).astype(numpy.float32)
    params = {"low": low, "high": high}
    sp_params = {"loc": low, "scale": high-low}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.random.normal(size=shape).astype(numpy.float32)
    return smp

tests = set(["batch_shape", "entropy", "event_shape", "log_prob", "mean",
             "sample", "support", "variance"])


@testing.distribution_unittest(distributions.Uniform, stats.uniform,
                               params_init, sample_for_test,
                               tests=tests, support="[low, high]")
class TestUniform(unittest.TestCase):
    pass
