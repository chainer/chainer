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
        self.dist = distributions.Poisson
        self.scipy_dist = stats.poisson

        self.test_targets = set([
            'batch_shape', 'event_shape', 'log_prob', 'mean', 'sample',
            'support', 'variance'])

        lam = numpy.random.uniform(0.1, 10, self.shape).astype(numpy.float32)
        self.params = {'lam': lam}
        self.scipy_params = {'mu': lam}

        self.continuous = False
        self.support = 'non negative integer'

    def sample_for_test(self):
        smp = numpy.random.randint(
            0, 10, self.sample_shape + self.shape).astype(numpy.int32)
        return smp


testing.run_module(__name__, __file__)
