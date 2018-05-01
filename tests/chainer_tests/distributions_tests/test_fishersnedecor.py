import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
}))
class TestFisherSnedecor(unittest.TestCase):
    def setUp(self):
        self.d1 = numpy.random.uniform(
            1, 7, self.shape).astype(numpy.float32)
        self.d2 = numpy.random.uniform(
            1, 7, self.shape).astype(numpy.float32)
        self.dist = distributions.FisherSnedecor(self.d1, self.d2)
        self.sp_dist = stats.f

    def test_batch_shape(self):
        self.assertEqual(self.dist.batch_shape, self.shape)

    def test_event_shape(self):
        self.assertEqual(self.dist.event_shape, ())

    def test_log_prob(self):
        smp = numpy.exp(
            numpy.random.normal(size=self.shape)).astype(numpy.float32)
        log_prob1 = self.dist.log_prob(smp).data
        log_prob2 = self.sp_dist.logpdf(smp, self.d1, self.d2)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_mean(self):
        mean1 = self.dist.mean.data
        mean2 = self.sp_dist.mean(self.d1, self.d2)
        testing.assert_allclose(mean1, mean2)

    def test_sample(self):
        smp1 = self.dist.sample(shape=(1000000)).data
        smp2 = self.sp_dist.rvs(self.d1, self.d2,
                                size=(1000000,)+self.shape)
        testing.assert_allclose(numpy.median(smp1, axis=0),
                                numpy.median(smp2, axis=0),
                                atol=1e-2, rtol=1e-2)

    def test_support(self):
        self.assertEqual(self.dist.support, "positive")

    def test_variance(self):
        variance1 = self.dist.variance.data
        variance2 = self.sp_dist.var(self.d1, self.d2)
        testing.assert_allclose(variance1, variance2)
