import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
}))
class TestNormal(unittest.TestCase):
    def setUp(self):
        self.lam = numpy.exp(numpy.random.uniform(
            -1, 1, self.shape)).astype(numpy.float32)
        self.dist = distributions.Exponential(self.lam)
        self.sp_dist = stats.expon

    def test_batch_shape(self):
        self.assertEqual(self.dist.batch_shape, self.shape)

    def test_cdf(self):
        smp = numpy.exp(numpy.random.normal(self.shape)).astype(numpy.float32)
        cdf1 = self.dist.cdf(smp).data
        cdf2 = self.sp_dist.cdf(smp, scale=1/self.lam)
        testing.assert_allclose(cdf1, cdf2)

    def test_entropy(self):
        ent1 = self.dist.entropy.data
        ent2 = self.sp_dist.entropy(scale=1/self.lam)
        testing.assert_allclose(ent1, ent2,
                                atol=1e-2, rtol=1e-2)

    def test_event_shape(self):
        self.assertEqual(self.dist.event_shape, ())

    def test_log_prob(self):
        smp = numpy.random.normal(self.shape).astype(numpy.float32)
        log_prob1 = self.dist.log_prob(smp).data
        log_prob2 = self.sp_dist.logpdf(smp, scale=1/self.lam)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_mean(self):
        mean1 = self.dist.mean.data
        mean2 = self.sp_dist.mean(scale=1/self.lam)
        testing.assert_allclose(mean1, mean2)

    def test_sample(self):
        smp1 = self.dist.sample(shape=(1000000)).data
        smp2 = self.sp_dist.rvs(scale=1/self.lam,
                                size=(1000000,)+self.shape)
        testing.assert_allclose(smp1.mean(axis=0), smp2.mean(axis=0),
                                atol=1e-2, rtol=1e-2)

    def test_support(self):
        self.assertEqual(self.dist.support, "positive")

    def test_variance(self):
        variance1 = self.dist.variance.data
        variance2 = self.sp_dist.var(scale=1/self.lam)
        testing.assert_allclose(variance1, variance2)
