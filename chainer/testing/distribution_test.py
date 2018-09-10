import unittest

import chainer
from chainer.backends import cuda
from chainer.testing import array
from chainer.testing import attr
import numpy


def skip_not_in_test_target(test_target):
    def decorator(f):
        def new_f(self, *args, **kwargs):
            if test_target not in self.test_targets:
                self.skipTest(
                    "\'%s\' is not exist in test_targets." % test_target)
            else:
                f(self, *args, **kwargs)
        return new_f
    return decorator


class distribution_unittest(unittest.TestCase):

    def setUp(self):
        self.support = 'real'
        self.event_shape = ()
        self.continuous = True
        self.scipy_onebyone = False
        self.test_targets = set()

        self.params_init()
        if self.is_variable:
            self.params = {k: chainer.Variable(v)
                           for k, v in self.params.items()}
        self.scipy_onebyone_params = \
            {k: v.reshape((numpy.prod(self.shape),)
                          + v.shape[len(self.shape):])
             for k, v in self.scipy_params.items()}

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

    @skip_not_in_test_target('batch_shape')
    def test_batch_shape_cpu(self):
        self.assertEqual(self.cpu_dist.batch_shape, self.shape)

    @skip_not_in_test_target('batch_shape')
    @attr.gpu
    def test_batch_shape_gpu(self):
        self.assertEqual(self.gpu_dist.batch_shape, self.shape)

    def check_cdf(self, is_gpu):
        smp = self.sample_for_test()
        if is_gpu:
            cdf1 = self.gpu_dist.cdf(cuda.to_gpu(smp)).data
        else:
            cdf1 = self.cpu_dist.cdf(smp).data
        cdf2 = self.scipy_dist.cdf(smp, **self.scipy_params)
        array.assert_allclose(cdf1, cdf2)

    @skip_not_in_test_target('cdf')
    def test_cdf_cpu(self):
        self.check_cdf(False)

    @skip_not_in_test_target('cdf')
    @attr.gpu
    def test_cdf_gpu(self):
        self.check_cdf(True)

    def check_entropy(self, is_gpu):
        if is_gpu:
            ent1 = self.gpu_dist.entropy.data
        else:
            ent1 = self.cpu_dist.entropy.data
        if self.scipy_onebyone:
            ent2 = []
            for i in range(numpy.prod(self.shape)):
                one_params = {k: v[i] for k, v
                              in self.scipy_onebyone_params.items()}
                ent2.append(self.scipy_dist.entropy(**one_params))
            ent2 = numpy.vstack(ent2).reshape(self.shape)
        else:
            ent2 = self.scipy_dist.entropy(**self.scipy_params)
        array.assert_allclose(ent1, ent2)

    @skip_not_in_test_target('entropy')
    def test_entropy_cpu(self):
        self.check_entropy(False)

    @skip_not_in_test_target('entropy')
    @attr.gpu
    def test_entropy_gpu(self):
        self.check_entropy(True)

    @skip_not_in_test_target('event_shape')
    def test_event_shape_cpu(self):
        self.assertEqual(self.cpu_dist.event_shape, self.event_shape)

    @skip_not_in_test_target('event_shape')
    @attr.gpu
    def test_event_shape_gpu(self):
        self.assertEqual(self.gpu_dist.event_shape, self.event_shape)

    def check_icdf(self, is_gpu):
        smp = numpy.random.uniform(
            1e-5, 1 - 1e-5, self.sample_shape + self.shape
        ).astype(numpy.float32)
        if is_gpu:
            icdf1 = self.gpu_dist.icdf(cuda.to_gpu(smp)).data
        else:
            icdf1 = self.cpu_dist.icdf(smp).data
        icdf2 = self.scipy_dist.ppf(smp, **self.scipy_params)
        array.assert_allclose(icdf1, icdf2)

    @skip_not_in_test_target('icdf')
    def test_icdf_cpu(self):
        self.check_icdf(False)

    @skip_not_in_test_target('icdf')
    @attr.gpu
    def test_icdf_gpu(self):
        self.check_icdf(True)

    def check_log_cdf(self, is_gpu):
        smp = self.sample_for_test()
        if is_gpu:
            log_cdf1 = self.gpu_dist.log_cdf(cuda.to_gpu(smp)).data
        else:
            log_cdf1 = self.cpu_dist.log_cdf(smp).data
        log_cdf2 = self.scipy_dist.logcdf(smp, **self.scipy_params)
        array.assert_allclose(log_cdf1, log_cdf2)

    @skip_not_in_test_target('log_cdf')
    def test_log_cdf_cpu(self):
        self.check_log_cdf(False)

    @skip_not_in_test_target('log_cdf')
    @attr.gpu
    def test_log_cdf_gpu(self):
        self.check_log_cdf(True)

    def check_log_prob(self, is_gpu):
        smp = self.sample_for_test()
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
                (int(numpy.prod(self.sample_shape)),
                 numpy.prod(self.shape),
                 int(numpy.prod(self.event_shape))))
            onebyone_smp = numpy.swapaxes(onebyone_smp, 0, 1)
            onebyone_smp = onebyone_smp.reshape((-1,) + self.sample_shape
                                                + self.event_shape)
            log_prob2 = []
            for i in range(numpy.prod(self.shape)):
                one_params = {k: v[i] for k, v
                              in self.scipy_onebyone_params.items()}
                one_smp = onebyone_smp[i]
                log_prob2.append(scipy_prob(one_smp, **one_params))
            log_prob2 = numpy.vstack(log_prob2)
            log_prob2 = log_prob2.reshape(numpy.prod(self.shape), -1).T
            log_prob2 = log_prob2.reshape(self.sample_shape + self.shape)
        else:
            log_prob2 = scipy_prob(smp, **self.scipy_params)
        array.assert_allclose(log_prob1, log_prob2)

    @skip_not_in_test_target('log_prob')
    def test_log_prob_cpu(self):
        self.check_log_prob(False)

    @skip_not_in_test_target('log_prob')
    @attr.gpu
    def test_log_prob_gpu(self):
        self.check_log_prob(True)

    def check_log_survival(self, is_gpu):
        smp = self.sample_for_test()
        if is_gpu:
            log_survival1 = \
                self.gpu_dist.log_survival_function(cuda.to_gpu(smp)).data
        else:
            log_survival1 = self.cpu_dist.log_survival_function(smp).data
        log_survival2 = self.scipy_dist.logsf(smp, **self.scipy_params)
        array.assert_allclose(log_survival1, log_survival2)

    @skip_not_in_test_target('log_survival')
    def test_log_survival_cpu(self):
        self.check_log_survival(False)

    @skip_not_in_test_target('log_survival')
    @attr.gpu
    def test_log_survival_gpu(self):
        self.check_log_survival(True)

    def check_mean(self, is_gpu):
        if is_gpu:
            mean1 = self.gpu_dist.mean.data
        else:
            mean1 = self.cpu_dist.mean.data

        if self.scipy_onebyone:
            mean2 = []
            for i in range(numpy.prod(self.shape)):
                one_params = {k: v[i] for k, v
                              in self.scipy_onebyone_params.items()}
                mean2.append(self.scipy_dist.mean(**one_params))
            mean2 = numpy.vstack(mean2).reshape(
                self.shape + self.cpu_dist.event_shape)
        else:
            mean2 = self.scipy_dist.mean(**self.scipy_params)
        array.assert_allclose(mean1, mean2)

    @skip_not_in_test_target('mean')
    def test_mean_cpu(self):
        self.check_mean(False)

    @skip_not_in_test_target('mean')
    @attr.gpu
    def test_mean_gpu(self):
        self.check_mean(True)

    def check_prob(self, is_gpu):
        smp = self.sample_for_test()
        if is_gpu:
            prob1 = self.gpu_dist.prob(cuda.to_gpu(smp)).data
        else:
            prob1 = self.cpu_dist.prob(smp).data
        if self.continuous:
            prob2 = self.scipy_dist.pdf(smp, **self.scipy_params)
        else:
            prob2 = self.scipy_dist.pmf(smp, **self.scipy_params)
        array.assert_allclose(prob1, prob2)

    @skip_not_in_test_target('prob')
    def test_prob_cpu(self):
        self.check_prob(False)

    @skip_not_in_test_target('prob')
    @attr.gpu
    def test_prob_gpu(self):
        self.check_prob(True)

    def check_sample(self, is_gpu):
        if is_gpu:
            smp1 = self.gpu_dist.sample(
                sample_shape=(100000,)+self.sample_shape).data
        else:
            smp1 = self.cpu_dist.sample(
                sample_shape=(100000,)+self.sample_shape).data

        if self.scipy_onebyone:
            smp2 = []
            for i in range(numpy.prod(self.shape)):
                one_params = {k: v[i] for k, v
                              in self.scipy_onebyone_params.items()}
                smp2.append(self.scipy_dist.rvs(
                    size=(100000,)+self.sample_shape, **one_params))
            smp2 = numpy.vstack(smp2)
            smp2 = smp2.reshape((numpy.prod(self.shape), 100000)
                                + self.sample_shape
                                + self.cpu_dist.event_shape)
            smp2 = numpy.rollaxis(
                smp2, 0, smp2.ndim-len(self.cpu_dist.event_shape))
            smp2 = smp2.reshape((100000,) + self.sample_shape + self.shape
                                + self.cpu_dist.event_shape)
        else:
            smp2 = self.scipy_dist.rvs(
                size=(100000,) + self.sample_shape + self.shape,
                **self.scipy_params)
        array.assert_allclose(smp1.mean(axis=0), smp2.mean(axis=0),
                              atol=3e-2, rtol=3e-2)
        array.assert_allclose(smp1.std(axis=0), smp2.std(axis=0),
                              atol=3e-2, rtol=3e-2)

    @skip_not_in_test_target('sample')
    def test_sample_cpu(self):
        self.check_sample(False)

    @skip_not_in_test_target('sample')
    @attr.gpu
    def test_sample_gpu(self):
        self.check_sample(True)

    def check_stddev(self, is_gpu):
        if is_gpu:
            stddev1 = self.gpu_dist.stddev.data
        else:
            stddev1 = self.cpu_dist.stddev.data
        stddev2 = self.scipy_dist.std(**self.scipy_params)
        array.assert_allclose(stddev1, stddev2)

    @skip_not_in_test_target('stddev')
    def test_stddev_cpu(self):
        self.check_stddev(False)

    @skip_not_in_test_target('stddev')
    @attr.gpu
    def test_stddev_gpu(self):
        self.check_stddev(True)

    @skip_not_in_test_target('support')
    def test_support_cpu(self):
        self.assertEqual(self.cpu_dist.support, self.support)

    @skip_not_in_test_target('support')
    @attr.gpu
    def test_support_gpu(self):
        self.assertEqual(self.gpu_dist.support, self.support)

    def check_survival(self, is_gpu):
        smp = self.sample_for_test()
        if is_gpu:
            survival1 = self.gpu_dist.survival_function(
                cuda.to_gpu(smp)).data
        else:
            survival1 = self.cpu_dist.survival_function(smp).data
        survival2 = self.scipy_dist.sf(smp, **self.scipy_params)
        array.assert_allclose(survival1, survival2)

    @skip_not_in_test_target('survival')
    def test_survival_cpu(self):
        self.check_survival(False)

    @skip_not_in_test_target('survival')
    @attr.gpu
    def test_survival_gpu(self):
        self.check_survival(True)

    def check_variance(self, is_gpu):
        if is_gpu:
            variance1 = self.gpu_dist.variance.data
        else:
            variance1 = self.cpu_dist.variance.data

        if self.scipy_onebyone:
            variance2 = []
            for i in range(numpy.prod(self.shape)):
                one_params = {k: v[i] for k, v
                              in self.scipy_onebyone_params.items()}
                variance2.append(self.scipy_dist.var(**one_params))
            variance2 = numpy.vstack(variance2).reshape(
                self.shape + self.cpu_dist.event_shape)
        else:
            variance2 = self.scipy_dist.var(**self.scipy_params)
        array.assert_allclose(variance1, variance2)

    @skip_not_in_test_target('variance')
    def test_variance_cpu(self):
        self.check_variance(False)

    @skip_not_in_test_target('variance')
    @attr.gpu
    def test_variance_gpu(self):
        self.check_variance(True)
