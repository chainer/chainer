import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
}))
class TestCauchy(unittest.TestCase):
    def setUp(self):
        self.loc = numpy.random.uniform(
            -1, 1, self.shape).astype(numpy.float32)
        self.scale = numpy.exp(numpy.random.uniform(
            -1, 1, self.shape)).astype(numpy.float32)
        self.dist = distributions.Cauchy(self.loc, self.scale)
        self.sp_dist = stats.cauchy

    def test_batch_shape(self):
        self.assertEqual(self.dist.batch_shape, self.shape)

    def test_cdf(self):
        smp = numpy.random.normal(self.shape).astype(numpy.float32)
        cdf1 = self.dist.cdf(smp).data
        cdf2 = self.sp_dist.cdf(smp, loc=self.loc, scale=self.scale)
        testing.assert_allclose(cdf1, cdf2)

    def test_entropy(self):
        ent1 = self.dist.entropy.data
        ent2 = self.sp_dist.entropy(self.loc, self.scale)
        testing.assert_allclose(ent1, ent2, atol=1e-2, rtol=1e-2)

    def test_event_shape(self):
        self.assertEqual(self.dist.event_shape, ())

    def test_icdf(self):
        smp = numpy.random.uniform(
            1e-5, 1, self.shape).astype(numpy.float32)
        icdf1 = self.dist.icdf(smp).data
        icdf2 = self.sp_dist.ppf(smp, loc=self.loc, scale=self.scale)
        testing.assert_allclose(icdf1, icdf2)

    def test_log_prob(self):
        smp = numpy.random.normal(self.shape).astype(numpy.float32)
        log_prob1 = self.dist.log_prob(smp).data
        log_prob2 = self.sp_dist.logpdf(smp, loc=self.loc, scale=self.scale)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_mean(self):
        mean1 = self.dist.mean
        mean2 = self.sp_dist.mean(self.loc, self.scale)
        testing.assert_allclose(mean1, mean2)

    def test_sample(self):
        smp1 = self.dist.sample(shape=(1000000)).data
        smp2 = self.sp_dist.rvs(loc=self.loc, scale=self.scale,
                                size=(1000000,)+self.shape)

        testing.assert_allclose(numpy.median(smp1, axis=0),
                                numpy.median(smp2, axis=0),
                                atol=1e-2, rtol=1e-2)

    def test_support(self):
        self.assertEqual(self.dist.support, "real")

    def test_variance(self):
        variance1 = self.dist.variance
        variance2 = self.sp_dist.var(loc=self.loc, scale=self.scale)
        testing.assert_allclose(variance1, variance2)

