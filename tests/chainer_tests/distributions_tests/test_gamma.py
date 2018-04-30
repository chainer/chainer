import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
}))
class TestGamma(unittest.TestCase):
    def setUp(self):
        self.k = numpy.random.uniform(
            0, 10, self.shape).astype(numpy.float32)
        self.theta = numpy.random.uniform(
            0, 10, self.shape).astype(numpy.float32)
        self.dist = distributions.Gamma(self.k, self.theta)
        self.sp_dist = stats.gamma

    def test_batch_shape(self):
        self.assertEqual(self.dist.batch_shape, self.shape)

    def test_entropy(self):
        ent1 = self.dist.entropy.data
        ent2 = self.sp_dist.entropy(
            self.k.reshape(-1),
            scale=self.theta.reshape(-1)).reshape(self.shape)
        testing.assert_allclose(ent1, ent2,
                                atol=1e-2, rtol=1e-2)

    def test_event_shape(self):
        self.assertEqual(self.dist.event_shape, ())

    def test_log_prob(self):
        smp = \
            numpy.random.gamma(shape=5., size=self.shape).astype(numpy.float32)
        log_prob1 = self.dist.log_prob(smp).data
        log_prob2 = self.sp_dist.logpdf(smp, a=self.k, scale=self.theta)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_mean(self):
        mean1 = self.dist.mean
        mean2 = self.sp_dist.mean(a=self.k, scale=self.theta)
        testing.assert_allclose(mean1, mean2)

    def test_sample(self):
        smp1 = self.dist.sample(shape=(1000000)).data
        smp2 = self.sp_dist.rvs(a=self.k, scale=self.theta,
                               size=(1000000,)+self.shape)
        testing.assert_allclose(smp1.mean(axis=0), smp2.mean(axis=0),
                                atol=1e-2, rtol=1e-2)
        testing.assert_allclose(smp1.std(axis=0), smp2.std(axis=0),
                                atol=1e-2, rtol=1e-2)

    def test_support(self):
        self.assertEqual(self.dist.support, "positive")

    def test_variance(self):
        variance1 = self.dist.variance
        variance2 = self.sp_dist.var(a=self.k, scale=self.theta)
        testing.assert_allclose(variance1, variance2)
