from chainer import distributions
from chainer import testing
from chainer import utils
import numpy


@testing.parameterize(*testing.product({
    'shape': [(2, 3), ()],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestGamma(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Gamma
        self.scipy_dist = stats.gamma

        self.test_targets = set(
            ['batch_shape', 'entropy', 'event_shape', 'log_prob', 'mean',
             'sample', 'support', 'variance'])

        k = utils.force_array(
            numpy.random.uniform(0, 5, self.shape).astype(numpy.float32))
        theta = utils.force_array(
            numpy.random.uniform(0, 5, self.shape).astype(numpy.float32))
        self.params = {'k': k, 'theta': theta}
        self.scipy_params = {'a': k, 'scale': theta}

        self.support = 'positive'

    def sample_for_test(self):
        smp = numpy.random.gamma(
            shape=5., size=self.sample_shape + self.shape
        ).astype(numpy.float32)
        return smp


testing.run_module(__name__, __file__)
