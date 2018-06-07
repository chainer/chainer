import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


def params_init(shape):
    loc = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
    scale = numpy.exp(numpy.random.uniform(-1, 0, shape)).astype(numpy.float32)
    params = {"loc": loc, "scale": scale}
    sp_params = {"s": scale, "scale": numpy.exp(loc)}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.random.lognormal(size=shape).astype(numpy.float32)
    return smp


tests = set(["batch_shape", "entropy", "event_shape", "log_prob", "mean",
             "sample", "support", "variance"])


@testing.distribution_unittest(distributions.LogNormal, stats.lognorm,
                               params_init, sample_for_test,
                               tests=tests, support="positive",
                               scipy_onebyone=True)
class TestLogNormal(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
