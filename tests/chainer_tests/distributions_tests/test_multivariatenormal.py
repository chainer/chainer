import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


def params_init(shape):
    loc = numpy.random.uniform(-1, 1, shape + (3,)).astype(numpy.float32)
    cov = numpy.random.normal(size=shape + (3, 3))
    cov = numpy.matmul(cov, numpy.rollaxis(cov, -1, -2)).astype(numpy.float32)
    l = numpy.linalg.cholesky(cov).astype(numpy.float32)
    params = {"loc": loc, "scale_tril": l}
    sp_params = {"mean": loc, "cov": cov}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.random.normal(size=shape + (3,)).astype(numpy.float32)
    return smp

tests = set(["batch_shape", "entropy", "event_shape", "log_prob",
             "support"])


@testing.distribution_unittest(distributions.MultivariateNormal,
                               stats.multivariate_normal,
                               params_init, sample_for_test,
                               tests=tests, scipy_onebyone=True,
                               event_shape=(3,))
class TestMultivariateNormal(unittest.TestCase):
    pass
