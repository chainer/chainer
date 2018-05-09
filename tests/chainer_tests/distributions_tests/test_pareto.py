import unittest

from chainer.backends import cuda
from chainer import distributions
from chainer import testing
from chainer.testing import attr
import numpy
from scipy import stats


def params_init(shape):
    scale = numpy.exp(numpy.random.uniform(-1, 1, shape)).astype(numpy.float32)
    alpha = numpy.exp(numpy.random.uniform(-1, 1, shape)).astype(numpy.float32)
    params = {"scale": scale, "alpha": alpha}
    sp_params = {"scale": scale, "b": alpha}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.random.pareto(a=1, size=shape).astype(numpy.float32)
    return smp

tests = set(["batch_shape", "entropy", "event_shape", "log_prob",
             "mean", "support", "variance"])


@testing.distribution_unittest(distributions.Pareto, stats.pareto,
                               params_init, sample_for_test,
                               tests=tests, support="[scale, inf]",
                               scipy_onebyone=True)
class TestPareto(unittest.TestCase):

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
