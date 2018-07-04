from chainer import distributions
from chainer import testing
import numpy


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
    'extreme_values': [True, False],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestBernoulli(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Bernoulli
        self.scipy_dist = stats.bernoulli

        self.test_targets = set([
            "batch_shape", "entropy", "log_prob", "mean", "prob", "sample",
            "stddev", "support", "variance"])

        if self.extreme_values:
            p = numpy.random.randint(0, 2, self.shape).astype(numpy.float32)
        else:
            p = numpy.random.uniform(0, 1, self.shape).astype(numpy.float32)

        self.params = {"p": p}
        self.scipy_params = {"p": p}

        self.support = '{0, 1}'
        self.continuous = False

    def sample_for_test(self):
        smp = numpy.random.randint(
            2, size=self.sample_shape + self.shape).astype(numpy.float32)
        return smp


testing.run_module(__name__, __file__)
