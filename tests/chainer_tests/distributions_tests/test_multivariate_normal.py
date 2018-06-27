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
class TestMultivariateNormal(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.MultivariateNormal
        self.scipy_dist = stats.multivariate_normal
        self.scipy_onebyone = True
        self.event_shape = (3,)

        self.test_targets = set([
            "batch_shape", "entropy", "event_shape", "log_prob",
            "support"])

        loc = numpy.random.uniform(
            -1, 1, self.shape + (3,)).astype(numpy.float32)
        cov = numpy.random.normal(size=self.shape + (3, 3))
        cov = numpy.matmul(
            cov, numpy.rollaxis(cov, -1, -2)).astype(numpy.float32)
        scale_tril = numpy.linalg.cholesky(cov).astype(numpy.float32)
        self.params = {"loc": loc, "scale_tril": scale_tril}
        self.scipy_params = {"mean": loc, "cov": cov}

    def sample_for_test(self):
        smp = numpy.random.normal(
            size=self.sample_shape + self.shape + (3,)).astype(numpy.float32)
        return smp


testing.run_module(__name__, __file__)
