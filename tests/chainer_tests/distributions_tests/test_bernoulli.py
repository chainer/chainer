import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
}))
class TestBernoulli(unittest.TestCase):
    def setUp(self):
        self.p = numpy.random.uniform(
            0, 1, self.shape).astype(numpy.float32)
        self.dist = distributions.Bernoulli(self.p)
        self.sp_dist = stats.bernoulli

    def test_batch_shape(self):
        self.assertEqual(self.dist.batch_shape, self.shape)

    def test_entropy(self):
        ent1 = self.dist.entropy.data
        ent2 = self.sp_dist.entropy(self.p)
        testing.assert_allclose(ent1, ent2,
                                atol=1e-2, rtol=1e-2)

    def test_enumerate_support(self):
        es = self.dist.enumerate_support.data
        testing.assert_allclose(
            es, numpy.broadcast_to(numpy.array([0., 1.]), es.shape))

    def test_event_shape(self):
        self.assertEqual(self.dist.event_shape, ())

    def test_log_prob(self):
        smp = numpy.random.randint(2, size=self.shape).astype(numpy.float32)
        log_prob1 = self.dist.log_prob(smp).data
        log_prob2 = self.sp_dist.logpmf(smp, p=self.p)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_mean(self):
        mean1 = self.dist.mean.data
        mean2 = self.sp_dist.mean(self.p)
        testing.assert_allclose(mean1, mean2)

    def test_mode(self):
        mode1 = self.dist.mode.data
        mode2 = numpy.round(self.p)
        testing.assert_allclose(mode1, mode2)

    def test_prob(self):
        smp = numpy.random.randint(2, size=self.shape).astype(numpy.float32)
        prob1 = self.dist.prob(smp).data
        prob2 = self.sp_dist.pmf(smp, p=self.p)
        testing.assert_allclose(prob1, prob2)

    def test_sample(self):
        smp1 = self.dist.sample(shape=(1000000)).data
        smp2 = self.sp_dist.rvs(p=self.p, size=(1000000,)+self.shape)
        testing.assert_allclose(smp1.mean(axis=0), smp2.mean(axis=0),
                                atol=1e-2, rtol=1e-2)

    def test_stddev(self):
        stddev1 = self.dist.stddev
        stddev2 = self.sp_dist.std(p=self.p)
        testing.assert_allclose(stddev1, stddev2)

    def test_variance(self):
        variance1 = self.dist.variance
        variance2 = self.sp_dist.var(p=self.p)
        testing.assert_allclose(variance1, variance2)
