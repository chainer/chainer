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
class TestChisquare(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Chisquare
        self.scipy_dist = stats.chi2

        self.test_targets = set([
            'batch_shape', 'entropy', 'event_shape', 'log_prob', 'mean',
            'sample', 'support', 'variance'])

        k = numpy.random.randint(1, 10, self.shape).astype(numpy.float32)
        self.params = {'k': k}
        self.scipy_params = {'df': k}

        self.support = 'positive'

    def sample_for_test(self):
        smp = numpy.random.chisquare(
            df=1, size=self.sample_shape + self.shape
        ).astype(numpy.float32)
        return smp


testing.run_module(__name__, __file__)
