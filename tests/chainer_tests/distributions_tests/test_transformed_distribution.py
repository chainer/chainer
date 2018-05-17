import unittest

import chainer
from chainer.backends import cuda
from chainer import distributions
from chainer.functions.math import exponential
from chainer import testing
from chainer.testing import attr
import numpy
from scipy import stats


def params_init(shape):
    loc = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
    scale = numpy.exp(numpy.random.uniform(-1, 0, shape)).astype(numpy.float32)
    params = {"loc": loc, "scale": scale}
    sp_params = {"s": scale, "scale": numpy.exp(loc)}
    return params, sp_params


def sample_for_test(shape):
    smp = numpy.random.lognormal(size=shape).astype(numpy.float32)
    return smp


tests = set(["cdf", "log_prob", "sample"])


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
    'is_variable': [True, False],
    'smp_shape': [(3, 2), ()],
}))
@testing.fix_random()
class TestLogTransformedDistribution(unittest.TestCase):
    def setUp(self):
        self.base_dist = distributions.Normal
        self.scipy_dist = stats.lognorm
        self.params, self.scipy_params = params_init(self.shape)
        if self.is_variable:
            self.params = {k: chainer.Variable(v)
                           for k, v in self.params.items()}
        self.scipy_onebyone_params = \
            {k: v.reshape((numpy.prod(self.shape),)
                          + v.shape[len(self.shape):])
             for k, v in self.scipy_params.items()}
        self.sample_for_test = sample_for_test
        self.transform = {"forward": exponential.exp,
                          "inv": exponential.log,
                          "inv_logdet_jac": lambda x: -exponential.log(x)}
        self.continuous = True
        self.scipy_onebyone = False

    @property
    def cpu_dist(self):
        return distributions.TransformedDistribution(
            self.base_dist(**self.params), **self.transform)

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
        return distributions.TransformedDistribution(
            self.base_dist(**self.gpu_params), **self.transform)

    def check_cdf(self, is_gpu):
        smp = self.sample_for_test(self.smp_shape + self.shape)
        if is_gpu:
            cdf1 = self.gpu_dist.cdf(cuda.to_gpu(smp)).data
        else:
            cdf1 = self.cpu_dist.cdf(smp).data
        cdf2 = self.scipy_dist.cdf(smp, **self.scipy_params)
        testing.assert_allclose(cdf1, cdf2)

    def test_cdf_cpu(self):
        self.check_cdf(False)

    @attr.gpu
    def test_cdf_gpu(self):
        self.check_cdf(True)

    def check_log_prob(self, is_gpu):
        smp = self.sample_for_test(self.smp_shape + self.shape)
        if is_gpu:
            log_prob1 = self.gpu_dist.log_prob(cuda.to_gpu(smp)).data
        else:
            log_prob1 = self.cpu_dist.log_prob(smp).data

        if self.continuous:
            scipy_prob = self.scipy_dist.logpdf
        else:
            scipy_prob = self.scipy_dist.logpmf

        if self.scipy_onebyone:
            onebyone_smp = smp.reshape(
                (int(numpy.prod(self.smp_shape)),
                 numpy.prod(self.shape),
                 int(numpy.prod(self.event_shape))))
            onebyone_smp = numpy.swapaxes(onebyone_smp, 0, 1)
            onebyone_smp = onebyone_smp.reshape((-1,) + self.smp_shape
                                                + self.event_shape)
            log_prob2 = []
            for i in range(numpy.prod(self.shape)):
                one_params = {k: v[i] for k, v
                              in self.scipy_onebyone_params.items()}
                one_smp = onebyone_smp[i]
                log_prob2.append(scipy_prob(one_smp, **one_params))
            log_prob2 = numpy.stack(log_prob2)
            log_prob2 = log_prob2.reshape(log_prob2.shape[0], -1).T
            log_prob2 = log_prob2.reshape(self.smp_shape + self.shape)
        else:
            log_prob2 = scipy_prob(smp, **self.scipy_params)
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

        if self.scipy_onebyone:
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
        else:
            smp2 = self.scipy_dist.rvs(
                size=(100000,) + self.smp_shape + self.shape,
                **self.scipy_params)
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
