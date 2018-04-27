import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
}))
class TestBeta(unittest.TestCase):
    def setUp(self):
        self.a = numpy.random.uniform(
            0, 10, self.shape).astype(numpy.float32)
        self.b = numpy.random.uniform(
            0, 10, self.shape).astype(numpy.float32)
        self.beta = distributions.Beta(self.a, self.b)

    def test_batch_shape(self):
        self.assertEqual(self.beta.batch_shape, self.shape)

    def test_entropy(self):
        ent1 = self.beta.entropy.data
        ent2 = stats.beta.entropy(
            self.a.reshape(-1), b=self.b.reshape(-1)).reshape(self.shape)
        testing.assert_allclose(ent1, ent2, atol=1e-2, rtol=1e-2)

    def test_event_shape(self):
        self.assertEqual(self.beta.event_shape, ())

    def test_log_prob(self):
        smp = \
            numpy.random.uniform(size=self.shape).astype(numpy.float32)
        log_prob1 = self.beta.log_prob(smp).data
        log_prob2 = stats.beta.logpdf(smp, a=self.a, b=self.b)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_mean(self):
        mean1 = self.beta.mean
        mean2 = stats.beta.mean(a=self.a, b=self.b)
        testing.assert_allclose(mean1, mean2)

    def test_sample(self):
        smp1 = self.beta.sample(shape=(1000000)).data
        smp2 = stats.beta.rvs(a=self.a, b=self.b,
                              size=(1000000,)+self.shape)
        testing.assert_allclose(smp1.mean(axis=0), smp2.mean(axis=0),
                                atol=1e-2, rtol=1e-2)
        testing.assert_allclose(smp1.std(axis=0), smp2.std(axis=0),
                                atol=1e-2, rtol=1e-2)

    def test_support(self):
        self.assertEqual(self.beta.support, "[0,1]")

    def test_variance(self):
        variance1 = self.beta.variance
        variance2 = stats.beta.var(a=self.a, b=self.b)
        testing.assert_allclose(variance1, variance2)
