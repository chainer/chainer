import numpy

from chainer import distributions
from chainer import testing


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
        self.dist = distributions.Exponential
        self.scipy_dist = stats.expon

        self.test_targets = set([
            'batch_shape', 'cdf', 'entropy', 'event_shape', 'icdf', 'log_prob',
            'mean', 'sample', 'support', 'variance'])

        lam = numpy.exp(numpy.random.uniform(
            -1, 1, self.shape)).astype(numpy.float32)
        lam = numpy.asarray(lam)
        self.params = {'lam': lam}
        self.scipy_params = {'scale': 1 / lam}

        self.support = 'positive'

    def sample_for_test(self):
        smp = numpy.exp(numpy.random.normal(
            size=self.sample_shape + self.shape)).astype(numpy.float32)
        return smp


testing.run_module(__name__, __file__)
