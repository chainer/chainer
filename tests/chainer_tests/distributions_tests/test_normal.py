import numpy

from chainer import distributions
from chainer import testing
from chainer import utils


@testing.parameterize(*testing.product({
    'shape': [(2, 3), ()],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
    'log_scale_option': [True, False],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestNormal(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Normal
        self.scipy_dist = stats.norm

        self.test_targets = set([
            'batch_shape', 'cdf', 'entropy', 'event_shape', 'icdf', 'log_cdf',
            'log_prob', 'log_survival', 'mean', 'prob', 'sample', 'stddev',
            'support', 'survival', 'variance'])

        loc = utils.force_array(
            numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32))
        if self.log_scale_option:
            log_scale = utils.force_array(
                numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32))
            scale = numpy.exp(log_scale)
            self.params = {'loc': loc, 'log_scale': log_scale}
            self.scipy_params = {'loc': loc, 'scale': scale}
        else:
            scale = utils.force_array(numpy.exp(
                numpy.random.uniform(-1, 1, self.shape)).astype(numpy.float32))
            self.params = {'loc': loc, 'scale': scale}
            self.scipy_params = {'loc': loc, 'scale': scale}

    def sample_for_test(self):
        smp = numpy.random.normal(
            size=self.sample_shape + self.shape).astype(numpy.float32)
        return smp


testing.run_module(__name__, __file__)
