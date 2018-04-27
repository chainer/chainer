import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
}))
class TestLaplace(unittest.TestCase):
    def setUp(self):
        self.loc = numpy.random.uniform(
            -1, 1, self.shape).astype(numpy.float32)
        self.scale = numpy.exp(numpy.random.uniform(
            -1, 1, self.shape)).astype(numpy.float32)
        self.laplace = distributions.Laplace(self.loc, self.scale)

    def test_batch_shape(self):
        self.assertEqual(self.laplace.batch_shape, self.shape)

    def test_cdf(self):
        smp = numpy.random.laplace(self.shape).astype(numpy.float32)
        cdf1 = self.laplace.cdf(smp).data
        cdf2 = stats.laplace.cdf(smp, loc=self.loc, scale=self.scale)
        testing.assert_allclose(cdf1, cdf2)

    def test_entropy(self):
        ent1 = self.laplace.entropy.data
        ent2 = stats.laplace.entropy(self.loc, self.scale)
        testing.assert_allclose(ent1, ent2,
                                atol=1e-2, rtol=1e-2)

    def test_event_shape(self):
        self.assertEqual(self.laplace.event_shape, ())

    def test_icdf(self):
        smp = numpy.random.uniform(
            1e-5, 1, self.shape).astype(numpy.float32)
        icdf1 = self.laplace.icdf(smp).data
        icdf2 = stats.laplace.ppf(smp, loc=self.loc, scale=self.scale)
        testing.assert_allclose(icdf1, icdf2)

    def test_log_prob(self):
        smp = numpy.random.laplace(self.shape).astype(numpy.float32)
        log_prob1 = self.laplace.log_prob(smp).data
        log_prob2 = stats.laplace.logpdf(smp, loc=self.loc, scale=self.scale)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_mean(self):
        mean1 = self.laplace.mean
        mean2 = stats.laplace.mean(self.loc, self.scale)
        testing.assert_allclose(mean1, mean2)

    def test_mode(self):
        mode1 = self.laplace.mode
        mode2 = stats.laplace.mean(self.loc, self.scale)
        testing.assert_allclose(mode1, mode2)

    def test_prob(self):
        smp = numpy.random.normal(self.shape).astype(numpy.float32)
        prob1 = self.laplace.prob(smp).data
        prob2 = stats.laplace.pdf(smp, loc=self.loc, scale=self.scale)
        testing.assert_allclose(prob1, prob2)

    def test_sample(self):
        smp1 = self.laplace.sample(shape=(1000000)).data
        smp2 = stats.laplace.rvs(loc=self.loc, scale=self.scale,
                                 size=(1000000,)+self.shape)
        testing.assert_allclose(smp1.mean(axis=0), smp2.mean(axis=0),
                                atol=1e-2, rtol=1e-2)

    def test_stddev(self):
        stddev1 = self.laplace.stddev
        stddev2 = stats.laplace.std(loc=self.loc, scale=self.scale)
        testing.assert_allclose(stddev1.data, stddev2)

    def test_support(self):
        self.assertEqual(self.laplace.support, "real")

    def test_variance(self):
        variance1 = self.laplace.variance
        variance2 = stats.laplace.var(loc=self.loc, scale=self.scale)
        testing.assert_allclose(variance1, variance2)
