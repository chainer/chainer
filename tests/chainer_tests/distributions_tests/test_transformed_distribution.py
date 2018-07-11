from chainer import distributions
from chainer import testing
import numpy


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestExpTransformedNormalDistribution(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats

        def dist(**params):
            return distributions.TransformedDistribution(
                distributions.Normal(**params), distributions.ExpBijector())

        self.dist = dist
        self.scipy_dist = stats.lognorm

        self.test_targets = set(["cdf", "log_prob", "prob", "sample"])

        loc = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        scale = numpy.exp(
            numpy.random.uniform(-1, 0, self.shape)).astype(numpy.float32)
        self.params = {"loc": loc, "scale": scale}
        self.scipy_params = {"s": scale, "scale": numpy.exp(loc)}

    def sample_for_test(self):
        smp = numpy.random.lognormal(
            size=self.sample_shape + self.shape).astype(numpy.float32)
        return smp


testing.run_module(__name__, __file__)
