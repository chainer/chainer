from chainer import distributions
from chainer import testing
import numpy


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
    'scipy_onebyone': [False, True],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestNormal(testing.distribution_unittest):

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Normal
        self.scipy_dist = stats.norm

        self.test_targets = set([
            "batch_shape", "cdf", "entropy", "event_shape", "icdf", "log_cdf",
            "log_prob", "log_survival", "mean", "prob", "sample", "stddev",
            "support", "survival", "variance"])

        loc = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        scale = numpy.exp(
            numpy.random.uniform(-1, 1, self.shape)).astype(numpy.float32)
        self.params = {"loc": loc, "scale": scale}
        self.scipy_params = {"loc": loc, "scale": scale}

    def sample_for_test(self):
        smp = numpy.random.normal(
            size=self.sample_shape + self.shape).astype(numpy.float32)
        return smp


testing.run_module(__name__, __file__)
