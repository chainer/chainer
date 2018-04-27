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
        self.gamma = distributions.Gamma(self.k, self.theta)

    def test_batch_shape(self):
        self.assertEqual(self.gamma.batch_shape, self.shape)

    def test_entropy(self):
        ent1 = self.gamma.entropy.data
        ent2 = stats.gamma.entropy(
            self.k.reshape(-1),
            scale=self.theta.reshape(-1)).reshape(self.shape)
        testing.assert_allclose(ent1, ent2,
                                atol=1e-2, rtol=1e-2)

    def test_event_shape(self):
        self.assertEqual(self.gamma.event_shape, ())

    def test_log_prob(self):
        smp = \
            numpy.random.gamma(shape=5., size=self.shape).astype(numpy.float32)
        log_prob1 = self.gamma.log_prob(smp).data
        log_prob2 = stats.gamma.logpdf(smp, a=self.k, scale=self.theta)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_mean(self):
        mean1 = self.gamma.mean
        mean2 = stats.gamma.mean(a=self.k, scale=self.theta)
        testing.assert_allclose(mean1, mean2)

    def test_sample(self):
        smp1 = self.gamma.sample(shape=(1000000)).data
        smp2 = stats.gamma.rvs(a=self.k, scale=self.theta,
                               size=(1000000,)+self.shape)
        testing.assert_allclose(smp1.mean(axis=0), smp2.mean(axis=0),
                                atol=1e-2, rtol=1e-2)
        testing.assert_allclose(smp1.std(axis=0), smp2.std(axis=0),
                                atol=1e-2, rtol=1e-2)

    def test_support(self):
        self.assertEqual(self.gamma.support, "positive")

    def test_variance(self):
        variance1 = self.gamma.variance
        variance2 = stats.gamma.var(a=self.k, scale=self.theta)
        testing.assert_allclose(variance1, variance2)
