import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
    'k': [3]
}))
class TestMultinomial(unittest.TestCase):
    def setUp(self):
        self.n = numpy.random.randint(1, 30, self.shape).astype(numpy.int32)
        self.p = numpy.random.normal(
            size=self.shape+(self.k,)).astype(numpy.float32)
        self.p = numpy.exp(self.p)
        self.p /= numpy.expand_dims(self.p.sum(axis=-1), axis=-1)
        self.dist = distributions.Multinomial(self.n, self.p)
        self.sp_dist = stats.multinomial

    def test_batch_shape(self):
        self.assertEqual(self.dist.batch_shape, self.shape)

    def test_event_shape(self):
        self.assertEqual(self.dist.event_shape, (self.k,))

    def test_log_prob(self):
        obo_p = self.p.reshape(-1, self.k)
        obo_n = self.n.reshape(-1)
        smp = [numpy.random.multinomial(one_n, one_p, size=1)
               for one_n, one_p in zip(obo_n, obo_p)]
        smp = numpy.stack(smp)
        smp = smp.reshape(self.shape + (-1,))

        log_prob1 = self.dist.log_prob(smp).data
        obo_smp = smp.reshape(-1, self.k)
        log_prob2 = [self.sp_dist.logpmf(one_smp, one_n, one_p)
                     for one_smp, one_n, one_p in zip(obo_smp, obo_n, obo_p)]
        log_prob2 = numpy.stack(log_prob2).reshape(self.shape)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_sample(self):
        smp = self.dist.sample(shape=(10000)).data
        smp_p = smp.mean(axis=0)
        testing.assert_allclose(self.p * numpy.expand_dims(self.n, axis=-1),
                                smp_p, atol=1e-2, rtol=1e-2)
