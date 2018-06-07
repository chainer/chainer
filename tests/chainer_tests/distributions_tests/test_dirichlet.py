import unittest

from chainer.backends import cuda
from chainer import distributions
from chainer import testing
from chainer.testing import attr
import numpy
from scipy import stats


def params_init(shape):
    alpha = numpy.random.uniform(0, 10, shape + (3,)).astype(numpy.float32)
    params = {"alpha": alpha}
    sp_params = {"alpha": alpha}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.random.normal(size=shape + (3,)).astype(numpy.float32)
    smp = numpy.exp(smp)
    smp /= numpy.expand_dims(smp.sum(axis=-1), axis=-1)
    return smp


tests = set(["batch_shape", "entropy", "event_shape", "mean", "sample",
             "support", "variance"])


@testing.distribution_unittest(distributions.Dirichlet, stats.dirichlet,
                               params_init, sample_for_test,
                               tests=tests, support="[0, 1]", event_shape=(3,),
                               scipy_onebyone=True)
class TestDirichlet(unittest.TestCase):

    def check_log_prob(self, is_gpu):
        smp = numpy.random.normal(size=self.smp_shape + self.shape + (3,))
        smp = numpy.exp(smp)
        smp /= numpy.expand_dims(smp.sum(axis=-1), axis=-1)
        if is_gpu:
            log_prob1 = self.gpu_dist.log_prob(
                cuda.to_gpu(smp.astype(numpy.float32))).data
        else:
            log_prob1 = self.cpu_dist.log_prob(smp.astype(numpy.float32)).data

        scipy_prob = self.scipy_dist.logpdf

        log_prob2 = []
        onebyone_smp = smp.reshape(
            (int(numpy.prod(self.smp_shape)), numpy.prod(self.shape), -1))
        onebyone_smp = numpy.swapaxes(onebyone_smp, 0, 1)
        onebyone_smp = numpy.swapaxes(onebyone_smp, 1, 2)
        for i in range(numpy.prod(self.shape)):
            one_params = {k: v[i] for k, v
                          in self.scipy_onebyone_params.items()}
            log_prob2.append(scipy_prob(onebyone_smp[i], **one_params))
        log_prob2 = numpy.stack(
            log_prob2).T.reshape(self.smp_shape + self.shape)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_log_prob_cpu(self):
        self.check_log_prob(False)

    @attr.gpu
    def test_log_prob_gpu(self):
        self.check_log_prob(True)


testing.run_module(__name__, __file__)
