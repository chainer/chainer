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
        self.loc = numpy.random.uniform(
            -1, 1, self.shape).astype(numpy.float32)
        self.scale = numpy.exp(numpy.random.uniform(
            -1, 1, self.shape)).astype(numpy.float32)
        self.normal = distributions.Normal(self.loc, self.scale)

    def test_batch_shape(self):
        self.assertEqual(self.normal.batch_shape, self.shape)

    def test_cdf(self):
        smp = numpy.random.normal(self.shape).astype(numpy.float32)
        cdf1 = self.normal.cdf(smp).data
        cdf2 = stats.norm.cdf(smp, loc=self.loc, scale=self.scale)
        testing.assert_allclose(cdf1, cdf2)

    def test_entropy(self):
        ent1 = self.normal.entropy.data
        ent2 = stats.norm.entropy(self.loc, self.scale)
        testing.assert_allclose(ent1, ent2,
                                atol=1e-2, rtol=1e-2)

    def test_event_shape(self):
        self.assertEqual(self.normal.event_shape, ())

    def test_icdf(self):
        smp = numpy.random.uniform(
            1e-5, 1, self.shape).astype(numpy.float32)
        icdf1 = self.normal.icdf(smp).data
        icdf2 = stats.norm.ppf(smp, loc=self.loc, scale=self.scale)
        testing.assert_allclose(icdf1, icdf2)

    def test_log_cdf(self):
        smp = numpy.random.normal(self.shape).astype(numpy.float32)
        log_cdf1 = self.normal.log_cdf(smp).data
        log_cdf2 = stats.norm.logcdf(smp, loc=self.loc, scale=self.scale)
        testing.assert_allclose(log_cdf1, log_cdf2)

    def test_log_prob(self):
        smp = numpy.random.normal(self.shape).astype(numpy.float32)
        log_prob1 = self.normal.log_prob(smp).data
        log_prob2 = stats.norm.logpdf(smp, loc=self.loc, scale=self.scale)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_log_survival(self):
        smp = numpy.random.normal(self.shape).astype(numpy.float32) / 10
        log_survival1 = self.normal.log_survival_function(smp).data
        log_survival2 = stats.norm.logsf(smp, loc=self.loc, scale=self.scale)
        testing.assert_allclose(log_survival1, log_survival2)

    def test_mean(self):
        mean1 = self.normal.mean
        mean2 = stats.norm.mean(self.loc, self.scale)
        testing.assert_allclose(mean1, mean2)

    def test_mode(self):
        mode1 = self.normal.mode
        mode2 = stats.norm.mean(self.loc, self.scale)
        testing.assert_allclose(mode1, mode2)

    def test_prob(self):
        smp = numpy.random.normal(self.shape).astype(numpy.float32)
        prob1 = self.normal.prob(smp).data
        prob2 = stats.norm.pdf(smp, loc=self.loc, scale=self.scale)
        testing.assert_allclose(prob1, prob2)

    def test_sample(self):
        smp1 = self.normal.sample(shape=(1000000)).data
        smp2 = stats.norm.rvs(loc=self.loc, scale=self.scale,
                              size=(1000000,)+self.shape)
        testing.assert_allclose(smp1.mean(axis=0), smp2.mean(axis=0),
                                atol=1e-2, rtol=1e-2)

    def test_stddev(self):
        stddev1 = self.normal.stddev
        stddev2 = stats.norm.std(loc=self.loc, scale=self.scale)
        testing.assert_allclose(stddev1, stddev2)

    def test_support(self):
        self.assertEqual(self.normal.support, "real")

    def test_survival(self):
        smp = numpy.random.normal(self.shape).astype(numpy.float32)
        survival1 = self.normal.survival_function(smp).data
        survival2 = stats.norm.sf(smp, loc=self.loc, scale=self.scale)
        testing.assert_allclose(survival1, survival2)

    def test_variance(self):
        variance1 = self.normal.variance
        variance2 = stats.norm.var(loc=self.loc, scale=self.scale)
        testing.assert_allclose(variance1, variance2)
