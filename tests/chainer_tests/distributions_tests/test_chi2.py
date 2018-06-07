import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


def params_init(shape):
    k = numpy.random.randint(1, 10, shape).astype(numpy.float32)
    params = {"k": k}
    sp_params = {"df": k}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.random.gamma(shape=5., size=shape).astype(numpy.float32)
    return smp


tests = set(["batch_shape", "entropy", "event_shape", "log_prob",
             "mean", "sample", "support", "variance"])


@testing.distribution_unittest(distributions.Chi2, stats.chi2,
                               params_init, sample_for_test,
                               tests=tests, support="positive")
class TestChi2(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
