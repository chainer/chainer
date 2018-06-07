import unittest

import chainer
from chainer.backends import cuda
from chainer import distributions
from chainer import testing
from chainer.testing import attr
import numpy
from scipy import stats


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
    'is_variable': [True, False],
    'smp_shape': [(3, 2), ()],
    'k': [3],
}))
class TestMultinomial(unittest.TestCase):
    def setUp(self):
        self.dist = distributions.Multinomial
        self.scipy_dist = stats.multinomial

        self.n = numpy.random.randint(1, 30, self.shape).astype(numpy.int32)
        self.p = numpy.random.normal(
            size=self.shape+(self.k,)).astype(numpy.float32)
        self.p = numpy.exp(self.p)
        self.p /= numpy.expand_dims(self.p.sum(axis=-1), axis=-1)

        self.params = {"n": self.n, "p": self.p}
        self.scipy_params = {"n": self.n, "p": self.p}
        if self.is_variable:
            self.params = {k: chainer.Variable(v)
                           for k, v in self.params.items()}
        self.scipy_onebyone_params = \
            {k: v.reshape(numpy.prod(self.shape), -1)
             for k, v in self.scipy_params.items()}
        self.event_shape = (self.k,)

    @property
    def cpu_dist(self):
        return self.dist(**self.params)

    @property
    def gpu_dist(self):
        if self.is_variable:
            self.gpu_params = {k: cuda.to_gpu(v.data)
                               for k, v in self.params.items()}
            self.gpu_params = {k: chainer.Variable(v)
                               for k, v in self.gpu_params.items()}
        else:
            self.gpu_params = {k: cuda.to_gpu(v)
                               for k, v in self.params.items()}
        return self.dist(**self.gpu_params)

    def test_batch_shape_cpu(self):
        self.assertEqual(self.cpu_dist.batch_shape, self.shape)

    @attr.gpu
    def test_batch_shape_gpu(self):
        self.assertEqual(self.gpu_dist.batch_shape, self.shape)

    def test_event_shape_cpu(self):
        self.assertEqual(self.cpu_dist.event_shape, self.event_shape)

    @attr.gpu
    def test_event_shape_gpu(self):
        self.assertEqual(self.gpu_dist.event_shape, self.event_shape)

    def check_log_prob(self, is_gpu):
        obo_p = self.p.reshape(-1, self.k)
        obo_n = self.n.reshape(-1)
        smp = [numpy.random.multinomial(one_n, one_p, size=1)
               for one_n, one_p in zip(obo_n, obo_p)]
        smp = numpy.stack(smp)
        smp = smp.reshape(self.shape + (-1,))
        if is_gpu:
            log_prob1 = self.gpu_dist.log_prob(cuda.to_gpu(smp)).data
        else:
            log_prob1 = self.cpu_dist.log_prob(smp).data

        obo_smp = smp.reshape(-1, self.k)
        log_prob2 = [self.scipy_dist.logpmf(one_smp, one_n, one_p)
                     for one_smp, one_n, one_p in zip(obo_smp, obo_n, obo_p)]
        log_prob2 = numpy.stack(log_prob2).reshape(self.shape)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_log_prob_cpu(self):
        self.check_log_prob(False)

    @attr.gpu
    def test_log_prob_gpu(self):
        self.check_log_prob(True)

    def check_sample(self, is_gpu):
        if is_gpu:
            smp1 = self.gpu_dist.sample(
                shape=(100000,)+self.smp_shape).data
        else:
            smp1 = self.cpu_dist.sample(
                shape=(100000,)+self.smp_shape).data

        smp2 = []
        for i in range(numpy.prod(self.shape)):
            one_params = {k: v[i] for k, v
                          in self.scipy_onebyone_params.items()}
            smp2.append(self.scipy_dist.rvs(
                size=(100000,)+self.smp_shape, **one_params))
        smp2 = numpy.stack(smp2)
        smp2 = numpy.rollaxis(
            smp2, 0, len(smp2.shape)-len(self.cpu_dist.event_shape))
        smp2 = smp2.reshape((100000,) + self.smp_shape + self.shape
                            + self.cpu_dist.event_shape)

        testing.assert_allclose(smp1.mean(axis=0), smp2.mean(axis=0),
                                atol=3e-2, rtol=3e-2)
        testing.assert_allclose(smp1.std(axis=0), smp2.std(axis=0),
                                atol=3e-2, rtol=3e-2)

    def test_sample_cpu(self):
        self.check_sample(False)

    @attr.gpu
    def test_sample_gpu(self):
        self.check_sample(True)


testing.run_module(__name__, __file__)
