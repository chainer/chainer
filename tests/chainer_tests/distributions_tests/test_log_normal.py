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
class TestNormal(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.LogNormal
        self.scipy_dist = stats.lognorm

        self.test_targets = set([
            "batch_shape", "entropy", "event_shape", "log_prob", "mean",
            "sample", "support", "variance"])

        loc = utils.force_array(
            numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32))
        scale = utils.force_array(numpy.exp(numpy.random.uniform(
            -1, 0, self.shape)).astype(numpy.float32))
        self.params = {"loc": loc, "scale": scale}
        self.scipy_params = {"s": scale, "scale": numpy.exp(loc)}

        self.support = 'positive'

    def sample_for_test(self):
        smp = numpy.random.lognormal(
            size=self.sample_shape + self.shape).astype(numpy.float32)
        return smp


testing.run_module(__name__, __file__)
