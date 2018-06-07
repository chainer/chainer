import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


def params_init(shape):
    n = numpy.random.randint(1, 30, shape).astype(numpy.int32)
    p = numpy.random.uniform(0, 1, shape).astype(numpy.float32)
    params = {"n": n, "p": p}
    sp_params = {"n": n, "p": p}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.random.randint(0, 10, shape).astype(numpy.int32)
    return smp


tests = set(["batch_shape", "event_shape", "log_prob", "mean", "sample",
             "support", "variance"])


@testing.distribution_unittest(distributions.Binomial, stats.binom,
                               params_init, sample_for_test,
                               tests=tests, continuous=False, support="[0, n]")
class TestBinomial(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
