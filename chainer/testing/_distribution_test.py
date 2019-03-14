import functools
import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer.testing import array
from chainer.testing import attr
from chainer import utils


def skip_not_in_test_target(test_target):
    def decorator(f):
        @functools.wraps(f)
        def new_f(self, *args, **kwargs):
            if test_target not in self.test_targets:
                self.skipTest(
                    '\'%s\' does not exist in test_targets.' % test_target)
            else:
                f(self, *args, **kwargs)
        return new_f
    return decorator


class distribution_unittest(unittest.TestCase):

    scipy_onebyone = False

    def setUp(self):
        self.support = 'real'
        if not hasattr(self, 'event_shape'):
            self.event_shape = ()
        self.continuous = True
        self.test_targets = set()
        self.options = {}

        self.setUp_configure()

        targets_not_found = self.test_targets - {
            'batch_shape', 'cdf', 'entropy', 'event_shape', 'icdf', 'log_cdf',
            'log_prob', 'log_survival', 'mean', 'prob', 'sample', 'stddev',
            'support', 'survival', 'variance'}
        if targets_not_found:
            raise ValueError(
                'invalid target(s): {}'.format(targets_not_found))

        if self.is_variable:
            self.params = {k: chainer.Variable(v)
                           for k, v in self.params.items()}

    def scipy_onebyone_params_iter(self):
        for index in numpy.ndindex(self.shape):
            yield {k: v[index] for k, v in self.scipy_params.items()}

    @property
    def cpu_dist(self):
        params = self.params
        params.update(self.options)
        return self.dist(**params)

    @property
    def gpu_dist(self):
        if self.is_variable:
            gpu_params = {k: cuda.to_gpu(v.data)
                          for k, v in self.params.items()}
            gpu_params = {k: chainer.Variable(v)
                          for k, v in gpu_params.items()}
        else:
            gpu_params = {k: cuda.to_gpu(v)
                          for k, v in self.params.items()}
        gpu_params.update(self.options)
        return self.dist(**gpu_params)

    @skip_not_in_test_target('batch_shape')
    def test_batch_shape_cpu(self):
        self.assertEqual(self.cpu_dist.batch_shape, self.shape)

    @attr.gpu
    @skip_not_in_test_target('batch_shape')
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

    @attr.gpu
    @skip_not_in_test_target('cdf')
    def test_cdf_gpu(self):
        self.check_cdf(True)

    def check_entropy(self, is_gpu):
        if is_gpu:
            ent1 = self.gpu_dist.entropy.data
        else:
            ent1 = self.cpu_dist.entropy.data
        if self.scipy_onebyone:
            ent2 = []
            for one_params in self.scipy_onebyone_params_iter():
                ent2.append(self.scipy_dist.entropy(**one_params))
            ent2 = numpy.vstack(ent2).reshape(self.shape)
        else:
            ent2 = self.scipy_dist.entropy(**self.scipy_params)
        array.assert_allclose(ent1, ent2)

    @skip_not_in_test_target('entropy')
    def test_entropy_cpu(self):
        self.check_entropy(False)

    @attr.gpu
    @skip_not_in_test_target('entropy')
    def test_entropy_gpu(self):
        self.check_entropy(True)

    @skip_not_in_test_target('event_shape')
    def test_event_shape_cpu(self):
        self.assertEqual(self.cpu_dist.event_shape, self.event_shape)

    @attr.gpu
    @skip_not_in_test_target('event_shape')
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

    @attr.gpu
    @skip_not_in_test_target('icdf')
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

    @attr.gpu
    @skip_not_in_test_target('log_cdf')
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
            onebyone_smp = smp.reshape(*[
                utils.size_of_shape(sh)
                for sh in [self.sample_shape, self.shape, self.event_shape]])
            onebyone_smp = numpy.swapaxes(onebyone_smp, 0, 1)
            onebyone_smp = onebyone_smp.reshape((-1,) + self.sample_shape
                                                + self.event_shape)
            log_prob2 = []
            for one_params, one_smp in zip(
                    self.scipy_onebyone_params_iter(), onebyone_smp):
                log_prob2.append(scipy_prob(one_smp, **one_params))
            log_prob2 = numpy.vstack(log_prob2)
            log_prob2 = log_prob2.reshape(
                utils.size_of_shape(self.shape), -1).T
            log_prob2 = log_prob2.reshape(self.sample_shape + self.shape)
        else:
            log_prob2 = scipy_prob(smp, **self.scipy_params)
        array.assert_allclose(log_prob1, log_prob2)

    @skip_not_in_test_target('log_prob')
    def test_log_prob_cpu(self):
        self.check_log_prob(False)

    @attr.gpu
    @skip_not_in_test_target('log_prob')
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

    @attr.gpu
    @skip_not_in_test_target('log_survival')
    def test_log_survival_gpu(self):
        self.check_log_survival(True)

    def check_mean(self, is_gpu):
        if is_gpu:
            mean1 = self.gpu_dist.mean.data
        else:
            mean1 = self.cpu_dist.mean.data

        if self.scipy_onebyone:
            mean2 = []
            for one_params in self.scipy_onebyone_params_iter():
                mean2.append(self.scipy_dist.mean(**one_params))
            mean2 = numpy.vstack(mean2).reshape(
                self.shape + self.cpu_dist.event_shape)
        else:
            mean2 = self.scipy_dist.mean(**self.scipy_params)
        array.assert_allclose(mean1, mean2)

    @skip_not_in_test_target('mean')
    def test_mean_cpu(self):
        self.check_mean(False)

    @attr.gpu
    @skip_not_in_test_target('mean')
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

    @attr.gpu
    @skip_not_in_test_target('prob')
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
            for one_params in self.scipy_onebyone_params_iter():
                smp2.append(self.scipy_dist.rvs(
                    size=(100000,)+self.sample_shape, **one_params))
            smp2 = numpy.vstack(smp2)
            smp2 = smp2.reshape((utils.size_of_shape(self.shape), 100000)
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

    @attr.gpu
    @skip_not_in_test_target('sample')
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

    @attr.gpu
    @skip_not_in_test_target('stddev')
    def test_stddev_gpu(self):
        self.check_stddev(True)

    @skip_not_in_test_target('support')
    def test_support_cpu(self):
        self.assertEqual(self.cpu_dist.support, self.support)

    @attr.gpu
    @skip_not_in_test_target('support')
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

    @attr.gpu
    @skip_not_in_test_target('survival')
    def test_survival_gpu(self):
        self.check_survival(True)

    def check_variance(self, is_gpu):
        if is_gpu:
            variance1 = self.gpu_dist.variance.data
        else:
            variance1 = self.cpu_dist.variance.data

        if self.scipy_onebyone:
            variance2 = []
            for one_params in self.scipy_onebyone_params_iter():
                variance2.append(self.scipy_dist.var(**one_params))
            variance2 = numpy.vstack(variance2).reshape(
                self.shape + self.cpu_dist.event_shape)
        else:
            variance2 = self.scipy_dist.var(**self.scipy_params)
        array.assert_allclose(variance1, variance2)

    @skip_not_in_test_target('variance')
    def test_variance_cpu(self):
        self.check_variance(False)

    @attr.gpu
    @skip_not_in_test_target('variance')
    def test_variance_gpu(self):
        self.check_variance(True)
