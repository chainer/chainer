import unittest

from chainer.backends import cuda
from chainer import distributions
from chainer import testing
from chainer.testing import attr
import numpy
from scipy import stats


def params_init(shape):
    d1 = numpy.random.uniform(1, 7, shape).astype(numpy.float32)
    d2 = numpy.random.uniform(1, 7, shape).astype(numpy.float32)
    params = {"d1": d1, "d2": d2}
    sp_params = {"dfn": d1, "dfd": d2}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.exp(numpy.random.normal(size=shape)).astype(numpy.float32)
    return smp

tests = set(["batch_shape", "event_shape", "log_prob", "mean",
             "support", "variance"])


@testing.distribution_unittest(distributions.FisherSnedecor, stats.f,
                               params_init, sample_for_test,
                               tests=tests, support="positive")
class TestFisherSnedecor(unittest.TestCase):

    def check_sample(self, is_gpu):
        if is_gpu:
            smp1 = self.gpu_dist.sample(
                shape=(100000,)+self.smp_shape).data
            smp1 = cuda.to_cpu(smp1)
        else:
            smp1 = self.cpu_dist.sample(
                shape=(100000,)+self.smp_shape).data
        smp2 = self.scipy_dist.rvs(
            size=(100000,)+self.smp_shape+self.shape, **self.scipy_params)
        testing.assert_allclose(numpy.median(smp1, axis=0),
                                numpy.median(smp2, axis=0),
                                atol=3e-2, rtol=3e-2)

    def test_sample_cpu(self):
        self.check_sample(False)

    @attr.gpu
    def test_sample_gpu(self):
        self.check_sample(True)
