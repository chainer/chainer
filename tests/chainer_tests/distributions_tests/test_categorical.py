import unittest

from chainer import distributions
from chainer import testing
import numpy
from scipy import stats


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
    'n': [3],
}))
class TestCategorical(unittest.TestCase):
    def setUp(self):
        self.p = numpy.random.normal(
            size=self.shape+(self.n,)).astype(numpy.float32)
        self.p = numpy.exp(self.p)
        self.p /= numpy.expand_dims(self.p.sum(axis=-1), axis=-1)
        self.dist = distributions.Categorical(self.p)
        self.sp_dist = stats.multinomial

    def test_batch_shape(self):
        self.assertEqual(self.dist.batch_shape, self.shape)

    def test_event_shape(self):
        self.assertEqual(self.dist.event_shape, ())

    def test_log_prob(self):
        smp = numpy.random.randint(0, self.n, self.shape).astype(numpy.int32)
        self.dist.log_prob(smp).data

    def test_sample(self):
        smp = self.dist.sample(shape=(10000)).data
        smp_p = [(smp == i).mean(axis=0) for i in range(self.n)]
        smp_p = numpy.array(smp_p)
        smp_p = numpy.rollaxis(smp_p, axis=0, start=len(smp_p.shape))
        testing.assert_allclose(self.p, smp_p,
                                atol=1e-2, rtol=1e-2)
