import unittest

import chainer
from chainer.backends import cuda
import numpy

try:
    from chainer.testing import attr
    _error = attr.get_error()
except ImportError as e:
    _error = e


def check_available():
    if _error is not None:
        raise RuntimeError('''\
{} is not available.

Reason: {}: {}'''.format(__name__, type(_error).__name__, _error))


def distribution_unittest(dist, scipy_dist_name, params_init, sample_for_test,
                          tests=set(), continuous=True, support="real",
                          event_shape=(), scipy_onebyone=False):
    check_available()

    from chainer import testing

    def f(klass):
        assert issubclass(klass, unittest.TestCase)

        def setUp(self):
            from scipy import stats
            self.dist = dist
            self.scipy_dist = stats.__dict__[scipy_dist_name]
            self.params, self.scipy_params = params_init(self.shape)
            if self.is_variable:
                self.params = {k: chainer.Variable(v)
                               for k, v in self.params.items()}
            self.scipy_onebyone_params = \
                {k: v.reshape((numpy.prod(self.shape),)
                              + v.shape[len(self.shape):])
                 for k, v in self.scipy_params.items()}
            self.sample_for_test = sample_for_test
            self.support = support
            self.event_shape = event_shape
            self.continuous = continuous
            self.scipy_onebyone = scipy_onebyone
        setattr(klass, "setUp", setUp)

        @property
        def cpu_dist(self):
            return self.dist(**self.params)
        setattr(klass, "cpu_dist", cpu_dist)

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
        setattr(klass, "gpu_dist", gpu_dist)

        def test_batch_shape_cpu(self):
            self.assertEqual(self.cpu_dist.batch_shape, self.shape)

        @attr.gpu
        def test_batch_shape_gpu(self):
            self.assertEqual(self.gpu_dist.batch_shape, self.shape)

        if "batch_shape" in tests:
            setattr(klass, "test_batch_shape_cpu", test_batch_shape_cpu)
            setattr(klass, "test_batch_shape_gpu", test_batch_shape_gpu)

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

        if "cdf" in tests:
            setattr(klass, "check_cdf", check_cdf)
            setattr(klass, "test_cdf_cpu", test_cdf_cpu)
            setattr(klass, "test_cdf_gpu", test_cdf_gpu)

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
                ent2 = numpy.stack(ent2).reshape(self.shape)
            else:
                ent2 = self.scipy_dist.entropy(**self.scipy_params)
            testing.assert_allclose(ent1, ent2)

        def test_entropy_cpu(self):
            self.check_entropy(False)

        @attr.gpu
        def test_entropy_gpu(self):
            self.check_entropy(True)

        if "entropy" in tests:
            setattr(klass, "check_entropy", check_entropy)
            setattr(klass, "test_entropy_cpu", test_entropy_cpu)
            setattr(klass, "test_entropy_gpu", test_entropy_gpu)

        def test_event_shape_cpu(self):
            self.assertEqual(self.cpu_dist.event_shape, self.event_shape)

        @attr.gpu
        def test_event_shape_gpu(self):
            self.assertEqual(self.gpu_dist.event_shape, self.event_shape)

        if "event_shape" in tests:
            setattr(klass, "test_event_shape_cpu", test_event_shape_cpu)
            setattr(klass, "test_event_shape_gpu", test_event_shape_gpu)

        def check_icdf(self, is_gpu):
            smp = numpy.random.uniform(
                1e-5, 1 - 1e-5, self.smp_shape + self.shape
            ).astype(numpy.float32)
            if is_gpu:
                icdf1 = self.gpu_dist.icdf(cuda.to_gpu(smp)).data
            else:
                icdf1 = self.cpu_dist.icdf(smp).data
            icdf2 = self.scipy_dist.ppf(smp, **self.scipy_params)
            testing.assert_allclose(icdf1, icdf2)

        def test_icdf_cpu(self):
            self.check_icdf(False)

        @attr.gpu
        def test_icdf_gpu(self):
            self.check_icdf(True)

        if "icdf" in tests:
            setattr(klass, "check_icdf", check_icdf)
            setattr(klass, "test_icdf_cpu", test_icdf_cpu)
            setattr(klass, "test_icdf_gpu", test_icdf_gpu)

        def check_log_cdf(self, is_gpu):
            smp = self.sample_for_test(self.smp_shape + self.shape)
            if is_gpu:
                log_cdf1 = self.gpu_dist.log_cdf(cuda.to_gpu(smp)).data
            else:
                log_cdf1 = self.cpu_dist.log_cdf(smp).data
            log_cdf2 = self.scipy_dist.logcdf(smp, **self.scipy_params)
            testing.assert_allclose(log_cdf1, log_cdf2)

        def test_log_cdf_cpu(self):
            self.check_log_cdf(False)

        @attr.gpu
        def test_log_cdf_gpu(self):
            self.check_log_cdf(True)

        if "log_cdf" in tests:
            setattr(klass, "check_log_cdf", check_log_cdf)
            setattr(klass, "test_log_cdf_cpu", test_log_cdf_cpu)
            setattr(klass, "test_log_cdf_gpu", test_log_cdf_gpu)

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

        if "log_prob" in tests:
            setattr(klass, "check_log_prob", check_log_prob)
            setattr(klass, "test_log_prob_cpu", test_log_prob_cpu)
            setattr(klass, "test_log_prob_gpu", test_log_prob_gpu)

        def check_log_survival(self, is_gpu):
            smp = self.sample_for_test(self.smp_shape + self.shape)
            if is_gpu:
                log_survival1 = \
                    self.gpu_dist.log_survival_function(cuda.to_gpu(smp)).data
            else:
                log_survival1 = self.cpu_dist.log_survival_function(smp).data
            log_survival2 = self.scipy_dist.logsf(smp, **self.scipy_params)
            testing.assert_allclose(log_survival1, log_survival2)

        def test_log_survival_cpu(self):
            self.check_log_survival(False)

        @attr.gpu
        def test_log_survival_gpu(self):
            self.check_log_survival(True)

        if "log_survival" in tests:
            setattr(klass, "check_log_survival", check_log_survival)
            setattr(klass, "test_log_survival_cpu", test_log_survival_cpu)
            setattr(klass, "test_log_survival_gpu", test_log_survival_gpu)

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
                mean2 = numpy.stack(mean2).reshape(
                    self.shape + self.cpu_dist.event_shape)
            else:
                mean2 = self.scipy_dist.mean(**self.scipy_params)
            testing.assert_allclose(mean1, mean2)

        def test_mean_cpu(self):
            self.check_mean(False)

        @attr.gpu
        def test_mean_gpu(self):
            self.check_mean(True)

        if "mean" in tests:
            setattr(klass, "check_mean", check_mean)
            setattr(klass, "test_mean_cpu", test_mean_cpu)
            setattr(klass, "test_mean_gpu", test_mean_gpu)

        def check_prob(self, is_gpu):
            smp = self.sample_for_test(self.smp_shape + self.shape)
            if is_gpu:
                prob1 = self.gpu_dist.prob(cuda.to_gpu(smp)).data
            else:
                prob1 = self.cpu_dist.prob(smp).data
            if self.continuous:
                prob2 = self.scipy_dist.pdf(smp, **self.scipy_params)
            else:
                prob2 = self.scipy_dist.pmf(smp, **self.scipy_params)
            testing.assert_allclose(prob1, prob2)

        def test_prob_cpu(self):
            self.check_prob(False)

        @attr.gpu
        def test_prob_gpu(self):
            self.check_prob(True)

        if "prob" in tests:
            setattr(klass, "check_prob", check_prob)
            setattr(klass, "test_prob_cpu", test_prob_cpu)
            setattr(klass, "test_prob_gpu", test_prob_gpu)

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

        if "sample" in tests:
            setattr(klass, "check_sample", check_sample)
            setattr(klass, "test_sample_cpu", test_sample_cpu)
            setattr(klass, "test_sample_gpu", test_sample_gpu)

        def check_stddev(self, is_gpu):
            if is_gpu:
                stddev1 = self.gpu_dist.stddev.data
            else:
                stddev1 = self.cpu_dist.stddev.data
            stddev2 = self.scipy_dist.std(**self.scipy_params)
            testing.assert_allclose(stddev1, stddev2)

        def test_stddev_cpu(self):
            self.check_stddev(False)

        @attr.gpu
        def test_stddev_gpu(self):
            self.check_stddev(True)

        if "stddev" in tests:
            setattr(klass, "check_stddev", check_stddev)
            setattr(klass, "test_stddev_cpu", test_stddev_cpu)
            setattr(klass, "test_stddev_gpu", test_stddev_gpu)

        def test_support_cpu(self):
            self.assertEqual(self.cpu_dist.support, self.support)

        @attr.gpu
        def test_support_gpu(self):
            self.assertEqual(self.gpu_dist.support, self.support)

        if "support" in tests:
            setattr(klass, "test_support_cpu", test_support_cpu)
            setattr(klass, "test_support_gpu", test_support_gpu)

        def check_survival(self, is_gpu):
            smp = self.sample_for_test(self.smp_shape + self.shape)
            if is_gpu:
                survival1 = self.gpu_dist.survival_function(
                    cuda.to_gpu(smp)).data
            else:
                survival1 = self.cpu_dist.survival_function(smp).data
            survival2 = self.scipy_dist.sf(smp, **self.scipy_params)
            testing.assert_allclose(survival1, survival2)

        def test_survival_cpu(self):
            self.check_survival(False)

        @attr.gpu
        def test_survival_gpu(self):
            self.check_survival(True)

        if "survival" in tests:
            setattr(klass, "check_survival", check_survival)
            setattr(klass, "test_survival_cpu", test_survival_cpu)
            setattr(klass, "test_survival_gpu", test_survival_gpu)

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
                variance2 = numpy.stack(variance2).reshape(
                    self.shape + self.cpu_dist.event_shape)
            else:
                variance2 = self.scipy_dist.var(**self.scipy_params)
            testing.assert_allclose(variance1, variance2)

        def test_variance_cpu(self):
            self.check_variance(False)

        @attr.gpu
        def test_variance_gpu(self):
            self.check_variance(True)

        if "variance" in tests:
            setattr(klass, "check_variance", check_variance)
            setattr(klass, "test_variance_cpu", test_variance_cpu)
            setattr(klass, "test_variance_gpu", test_variance_gpu)

        # Return parameterized class.
        return testing.parameterize(*testing.product({
            'shape': [(3, 2), (1,)],
            'is_variable': [True, False],
            'smp_shape': [(3, 2), ()],
        }))(testing.with_requires('scipy')(testing.fix_random()(klass)))
    return f
