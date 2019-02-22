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
class TestSkellam(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Skellam
        self.scipy_dist = stats.skellam

        self.test_targets = set([
            "batch_shape", "event_shape", "mean",
            "support", "variance", "cdf", "log_cdf"])

        mu1 = numpy.random.randint(1, 10, self.shape).astype(numpy.float32)
        mu2 = numpy.random.randint(1, 10, self.shape).astype(numpy.float32)
        self.params = {"mu1": mu1, "mu2": mu2}
        self.scipy_params = {"mu1": mu1, "mu2": mu2}

        self.support = 'positive integer'
        self.continuous = False

    def sample_for_test(self):
        smp = numpy.random.randint(
            1, 10, self.sample_shape + self.shape).astype(numpy.int32)
        return smp


testing.run_module(__name__, __file__)
