import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
    'k': [3],
}))
class TestDirichlet(unittest.TestCase):
    def setUp(self):
        self.alpha = numpy.random.uniform(
            0, 10, self.shape + (self.k,)).astype(numpy.float32)
        self.dist = distributions.Dirichlet(self.alpha)
        self.sp_dist = stats.dirichlet

    def test_batch_shape(self):
        self.assertEqual(self.dist.batch_shape, self.shape)

    def test_entropy(self):
        ent1 = self.dist.entropy.data
        obo_alpha = self.alpha.reshape(-1, self.alpha.shape[-1])
        ent2 = [self.sp_dist.entropy(one_alpha) for one_alpha in obo_alpha]
        ent2 = numpy.stack(ent2).reshape(self.shape)
        testing.assert_allclose(ent1, ent2, atol=1e-2, rtol=1e-2)

    def test_event_shape(self):
        self.assertEqual(self.dist.event_shape, (self.k,))

    def test_log_prob(self):
        smp = numpy.random.normal(
            size=self.shape + (self.k,))
        smp = numpy.exp(smp)
        smp /= numpy.expand_dims(smp.sum(axis=-1), axis=-1)
        log_prob1 = self.dist.log_prob(smp.astype(numpy.float32)).data

        obo_alpha = self.alpha.reshape(-1, self.alpha.shape[-1])
        obo_smp = smp.reshape(-1, self.k)
        log_prob2 = [self.sp_dist.logpdf(one_smp, one_alpha)
                     for one_alpha, one_smp in zip(obo_alpha, obo_smp)]
        log_prob2 = numpy.stack(log_prob2).reshape(self.shape)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_mean(self):
        mean1 = self.dist.mean.data
        obo_alpha = self.alpha.reshape(-1, self.alpha.shape[-1])
        mean2 = [self.sp_dist.mean(one_alpha) for one_alpha in obo_alpha]
        mean2 = numpy.stack(mean2).reshape(self.alpha.shape)
        testing.assert_allclose(mean1, mean2)

    def test_sample(self):
        smp1 = self.dist.sample(shape=(100000)).data
        obo_alpha = self.alpha.reshape(-1, self.alpha.shape[-1])
        smp2 = [self.sp_dist.rvs(one_alpha, size=100000)
                for one_alpha in obo_alpha]
        smp2 = numpy.stack(smp2).reshape((100000,) + self.alpha.shape)
        testing.assert_allclose(smp1.mean(axis=0), smp2.mean(axis=0),
                                atol=1e-2, rtol=1e-2)
        testing.assert_allclose(smp1.std(axis=0), smp2.std(axis=0),
                                atol=1e-2, rtol=1e-2)

    def test_support(self):
        self.assertEqual(self.dist.support, "[0,1]")

    def test_variance(self):
        variance1 = self.dist.variance.data
        obo_alpha = self.alpha.reshape(-1, self.alpha.shape[-1])
        variance2 = [self.sp_dist.var(one_alpha) for one_alpha in obo_alpha]
        variance2 = numpy.stack(variance2).reshape(self.alpha.shape)
        testing.assert_allclose(variance1, variance2)
