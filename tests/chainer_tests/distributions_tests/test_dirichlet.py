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
class TestDirichlet(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Dirichlet
        self.scipy_dist = stats.dirichlet

        self.test_targets = set([
            'batch_shape', 'entropy', 'event_shape', 'mean', 'sample',
            'support', 'variance'])

        alpha = numpy.random.uniform(
            0, 10, self.shape + (3,)).astype(numpy.float32)
        self.params = {'alpha': alpha}
        self.scipy_params = {'alpha': alpha}
        self.support = '[0, 1]'
        self.event_shape = (3,)

    def sample_for_test(self):
        smp = numpy.random.normal(size=self.shape + (3,)).astype(numpy.float32)
        smp = numpy.exp(smp)
        smp /= numpy.expand_dims(smp.sum(axis=-1), axis=-1)
        return smp


testing.run_module(__name__, __file__)
