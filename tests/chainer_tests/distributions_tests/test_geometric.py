from chainer import distributions
from chainer import testing
import numpy


@testing.parameterize(*testing.product({
    'shape': [(2, 3), ()],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestLogNormal(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Geometric
        self.scipy_dist = stats.geom

        self.test_targets = set([
            "batch_shape", "event_shape", "log_prob", "mean", "sample",
            "support", "variance"])

        p = numpy.random.uniform(0, 1, self.shape).astype(numpy.float32)
        self.params = {"p": p}
        self.scipy_params = {"p": p}

        self.support = 'positive integer'
        self.continuous = False

    def sample_for_test(self):
        smp = numpy.random.randint(
            1, 10, self.sample_shape + self.shape).astype(numpy.int32)
        return smp


testing.run_module(__name__, __file__)


'''
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


testing.run_module(__name__, __file__)
'''