import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
    'd': [3],
}))
class TestMultivariateNormal(unittest.TestCase):
    def setUp(self):
        self.loc = numpy.random.uniform(
            -1, 1, self.shape + (self.d,)).astype(numpy.float32)
        self.cov = numpy.random.normal(
            size=self.shape + (self.d, self.d))
        self.cov = numpy.matmul(
            self.cov, numpy.rollaxis(self.cov, -1, -2)).astype(numpy.float32)
        self.l = numpy.linalg.cholesky(self.cov).astype(numpy.float32)
        self.dist = distributions.MultivariateNormal(self.loc, self.l)
        self.sp_dist = stats.multivariate_normal

    def test_batch_shape(self):
        self.assertEqual(self.dist.batch_shape, self.shape)

    def test_entropy(self):
        ent1 = self.dist.entropy.data
        obo_loc = self.loc.reshape(-1, self.d)
        obo_cov = self.cov.reshape(-1, self.d, self.d)
        ent2 = [self.sp_dist.entropy(one_loc, one_cov)
                for one_loc, one_cov in zip(obo_loc, obo_cov)]
        ent2 = numpy.stack(ent2).reshape(self.shape)

        testing.assert_allclose(ent1, ent2,
                                atol=1e-2, rtol=1e-2)

    def test_event_shape(self):
        self.assertEqual(self.dist.event_shape, (self.d,))

    def test_log_prob(self):
        smp = numpy.random.normal(
            size=self.shape + (self.d,)).astype(numpy.float32)
        log_prob1 = self.dist.log_prob(smp).data

        obo_smp = smp.reshape(-1, self.d)
        obo_loc = self.loc.reshape(-1, self.d)
        obo_cov = self.cov.reshape(-1, self.d, self.d)
        log_prob2 = [self.sp_dist.logpdf(one_smp, one_loc, one_cov)
                     for one_smp, one_loc, one_cov
                     in zip(obo_smp, obo_loc, obo_cov)]
        log_prob2 = numpy.stack(log_prob2).reshape(self.shape)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_mean(self):
        mean1 = self.dist.mean.data
        mean2 = self.loc
        testing.assert_allclose(mean1, mean2)

    def test_mode(self):
        mode1 = self.dist.mode.data
        mode2 = self.loc
        testing.assert_allclose(mode1, mode2)

    def test_sample(self):
        smp1 = self.dist.sample(shape=(1000000)).data
        obo_loc = self.loc.reshape(-1, self.d)
        obo_cov = self.cov.reshape(-1, self.d, self.d)
        smp2 = [self.sp_dist.rvs(one_loc, one_cov, size=1000000)
                for one_loc, one_cov in zip(obo_loc, obo_cov)]
        smp2 = numpy.stack(smp2).reshape(self.shape + (1000000, self.d))
        testing.assert_allclose(smp1.mean(axis=0), smp2.mean(axis=-2),
                                atol=1e-2, rtol=1e-2)

    def test_support(self):
        self.assertEqual(self.dist.support, "real")

    def test_variance(self):
        self.dist.variance.data
