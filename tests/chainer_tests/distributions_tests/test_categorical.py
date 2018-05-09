import unittest

from chainer.backends import cuda
from chainer import distributions
from chainer import testing
import numpy
from scipy import stats

from chainer.testing import attr


def params_init(shape):
    p = numpy.random.normal(size=shape+(3,)).astype(numpy.float32)
    p = numpy.exp(p)
    p /= numpy.expand_dims(p.sum(axis=-1), axis=-1)
    params = {"p": p}
    sp_params = {"n": 1, "p": p}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.random.randint(0, 3, shape).astype(numpy.int32)
    return smp


tests = set(["batch_shape", "event_shape"])


@testing.distribution_unittest(distributions.Categorical, stats.multinomial,
                               params_init, sample_for_test,
                               tests=tests, continuous=False, support="[0, n]")
class TestCategorical(unittest.TestCase):

    def check_log_prob(self, is_gpu):
        smp = self.sample_for_test(self.smp_shape + self.shape)
        if is_gpu:
            log_prob1 = self.gpu_dist.log_prob(cuda.to_gpu(smp)).data
        else:
            log_prob1 = self.cpu_dist.log_prob(smp).data

        smp = numpy.eye(3)[smp]
        log_prob2 = self.scipy_dist.logpmf(smp, **self.scipy_params)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_log_prob_cpu(self):
        self.check_log_prob(False)

    @attr.gpu
    def test_log_prob_gpu(self):
        self.check_log_prob(True)

    def check_sample(self, is_gpu):
        if is_gpu:
            smp = self.gpu_dist.sample(shape=(10000)).data
            smp = cuda.to_cpu(smp)
        else:
            smp = self.cpu_dist.sample(shape=(10000)).data
        smp_p = [(smp == i).mean(axis=0) for i in range(3)]
        smp_p = numpy.array(smp_p)
        smp_p = numpy.rollaxis(smp_p, axis=0, start=len(smp_p.shape))
        testing.assert_allclose(smp_p, self.scipy_params["p"], atol=1e-2,
                                rtol=1e-2)

    def test_sample_cpu(self):
        self.check_sample(False)

    @attr.gpu
    def test_sample_gpu(self):
        self.check_sample(True)
