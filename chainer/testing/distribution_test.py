import unittest

from chainer.backends import cuda
from chainer import testing
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


def distribution_unittest(dist, scipy_dist, params_init, sample_for_test,
                          support="real", event_shape=(), ignore_tests=set()):
    check_available()

    def f(klass):
        assert issubclass(klass, unittest.TestCase)

        def setUp(self):
            self.dist = dist
            self.scipy_dist = scipy_dist
            self.params, self.scipy_params = params_init(self.shape)
            self.gpu_params = {k: cuda.to_gpu(v)
                               for k, v in self.params.items()}
            self.sample_for_test = sample_for_test
            self.support = support
            self.event_shape = event_shape
            self.ignore_tests = ignore_tests
        setattr(klass, "setUp", setUp)

        @property
        def cpu_dist(self):
            return self.dist(**self.params)
        setattr(klass, "cpu_dist", cpu_dist)

        @property
        def gpu_dist(self):
            return self.dist(**self.gpu_params)
        setattr(klass, "gpu_dist", gpu_dist)

        def test_batch_shape_cpu(self):
            self.assertEqual(self.cpu_dist.batch_shape, self.shape)
        setattr(klass, "test_batch_shape_cpu", test_batch_shape_cpu)

        @attr.gpu
        def test_batch_shape_gpu(self):
            self.assertEqual(self.gpu_dist.batch_shape, self.shape)
        setattr(klass, "test_batch_shape_gpu", test_batch_shape_gpu)

        def test_cdf_cpu(self):
            smp = self.sample_for_test(self.shape)
            cdf1 = self.cpu_dist.cdf(smp).data
            cdf2 = self.scipy_dist.cdf(smp, **self.scipy_params)
            testing.assert_allclose(cdf1, cdf2)
        setattr(klass, "test_cdf_cpu", test_cdf_cpu)

        @attr.gpu
        def test_cdf_gpu(self):
            smp = self.sample_for_test(self.shape)
            cdf1 = self.gpu_dist.cdf(cuda.to_gpu(smp)).data
            cdf2 = self.scipy_dist.cdf(smp, **self.scipy_params)
            testing.assert_allclose(cdf1, cdf2)
        setattr(klass, "test_cdf_gpu", test_cdf_gpu)

        def test_entropy_cpu(self):
            ent1 = self.cpu_dist.entropy.data
            ent2 = self.scipy_dist.entropy(**self.scipy_params)
            testing.assert_allclose(ent1, ent2)
        setattr(klass, "test_entropy_cpu", test_entropy_cpu)

        @attr.gpu
        def test_entropy_gpu(self):
            ent1 = self.gpu_dist.entropy.data
            ent2 = self.scipy_dist.entropy(**self.scipy_params)
            testing.assert_allclose(ent1, ent2)
        setattr(klass, "test_entropy_gpu", test_entropy_gpu)

        def test_event_shape_cpu(self):
            self.assertEqual(self.cpu_dist.event_shape, self.event_shape)
        setattr(klass, "test_event_shape_cpu", test_event_shape_cpu)

        @attr.gpu
        def test_event_shape_gpu(self):
            self.assertEqual(self.gpu_dist.event_shape, self.event_shape)
        setattr(klass, "test_event_shape_gpu", test_event_shape_gpu)

        def test_icdf_cpu(self):
            smp = numpy.random.uniform(
                1e-5, 1 - 1e-5, self.shape).astype(numpy.float32)
            icdf1 = self.cpu_dist.icdf(smp).data
            icdf2 = self.scipy_dist.ppf(smp, **self.scipy_params)
            testing.assert_allclose(icdf1, icdf2)
        setattr(klass, "test_icdf_cpu", test_icdf_cpu)

        @attr.gpu
        def test_icdf_gpu(self):
            smp = numpy.random.uniform(
                1e-5, 1 - 1e-5, self.shape).astype(numpy.float32)
            icdf1 = self.gpu_dist.icdf(cuda.to_gpu(smp)).data
            icdf2 = self.scipy_dist.ppf(smp, **self.scipy_params)
            testing.assert_allclose(icdf1, icdf2)
        setattr(klass, "test_icdf_gpu", test_icdf_gpu)

        def test_log_cdf_cpu(self):
            smp = self.sample_for_test(self.shape)
            log_cdf1 = self.cpu_dist.log_cdf(smp).data
            log_cdf2 = self.scipy_dist.logcdf(smp, **self.scipy_params)
            testing.assert_allclose(log_cdf1, log_cdf2)
        setattr(klass, "test_log_cdf_cpu", test_log_cdf_cpu)

        @attr.gpu
        def test_log_cdf_gpu(self):
            smp = self.sample_for_test(self.shape)
            log_cdf1 = self.gpu_dist.log_cdf(cuda.to_gpu(smp)).data
            log_cdf2 = self.scipy_dist.logcdf(smp, **self.scipy_params)
            testing.assert_allclose(log_cdf1, log_cdf2)
        setattr(klass, "test_log_cdf_gpu", test_log_cdf_gpu)

        def test_log_prob_cpu(self):
            smp = self.sample_for_test(self.shape)
            log_prob1 = self.cpu_dist.log_prob(smp).data
            log_prob2 = self.scipy_dist.logpdf(smp, **self.scipy_params)
            testing.assert_allclose(log_prob1, log_prob2)
        setattr(klass, "test_log_prob_cpu", test_log_prob_cpu)

        @attr.gpu
        def test_log_prob_gpu(self):
            smp = self.sample_for_test(self.shape)
            log_prob1 = self.gpu_dist.log_prob(cuda.to_gpu(smp)).data
            log_prob2 = self.scipy_dist.logpdf(smp, **self.scipy_params)
            testing.assert_allclose(log_prob1, log_prob2)
        setattr(klass, "test_log_prob_gpu", test_log_prob_gpu)

        def test_log_survival_cpu(self):
            smp = self.sample_for_test(self.shape)
            log_survival1 = self.cpu_dist.log_survival_function(smp).data
            log_survival2 = self.scipy_dist.logsf(smp, **self.scipy_params)
            testing.assert_allclose(log_survival1, log_survival2)
        setattr(klass, "test_log_survival_cpu", test_log_survival_cpu)

        @attr.gpu
        def test_log_survival_gpu(self):
            smp = self.sample_for_test(self.shape)
            log_survival1 = \
                self.gpu_dist.log_survival_function(cuda.to_gpu(smp)).data
            log_survival2 = self.scipy_dist.logsf(smp, **self.scipy_params)
            testing.assert_allclose(log_survival1, log_survival2)
        setattr(klass, "test_log_survival_gpu", test_log_survival_gpu)

        def test_mean_cpu(self):
            mean1 = self.cpu_dist.mean.data
            mean2 = self.scipy_dist.mean(**self.scipy_params)
            testing.assert_allclose(mean1, mean2)
        setattr(klass, "test_mean_cpu", test_mean_cpu)

        @attr.gpu
        def test_mean_gpu(self):
            mean1 = self.gpu_dist.mean.data
            mean2 = self.scipy_dist.mean(**self.scipy_params)
            testing.assert_allclose(mean1, mean2)
        setattr(klass, "test_mean_gpu", test_mean_gpu)

        def test_prob_cpu(self):
            smp = self.sample_for_test(self.shape)
            prob1 = self.cpu_dist.prob(smp).data
            prob2 = self.scipy_dist.pdf(smp, **self.scipy_params)
            testing.assert_allclose(prob1, prob2)
        setattr(klass, "test_prob_cpu", test_prob_cpu)

        @attr.gpu
        def test_prob_gpu(self):
            smp = self.sample_for_test(self.shape)
            prob1 = self.gpu_dist.prob(cuda.to_gpu(smp)).data
            prob2 = self.scipy_dist.pdf(smp, **self.scipy_params)
            testing.assert_allclose(prob1, prob2)
        setattr(klass, "test_prob_gpu", test_prob_gpu)

        def test_sample_cpu(self):
            smp1 = self.cpu_dist.sample(shape=(100000)).data
            smp2 = self.scipy_dist.rvs(**self.scipy_params,
                                       size=(100000,)+self.shape)
            testing.assert_allclose(smp1.mean(axis=0), smp2.mean(axis=0),
                                    atol=1e-2, rtol=1e-2)
            testing.assert_allclose(smp1.std(axis=0), smp2.std(axis=0),
                                    atol=1e-2, rtol=1e-2)
        setattr(klass, "test_sample_cpu", test_sample_cpu)

        @attr.gpu
        def test_sample_gpu(self):
            smp1 = self.gpu_dist.sample(shape=(100000)).data
            smp2 = self.scipy_dist.rvs(**self.scipy_params,
                                       size=(100000,)+self.shape)
            testing.assert_allclose(smp1.mean(axis=0), smp2.mean(axis=0),
                                    atol=1e-2, rtol=1e-2)
            testing.assert_allclose(smp1.std(axis=0), smp2.std(axis=0),
                                    atol=1e-2, rtol=1e-2)
        setattr(klass, "test_sample_gpu", test_sample_gpu)

        def test_stddev_cpu(self):
            stddev1 = self.cpu_dist.stddev.data
            stddev2 = self.scipy_dist.std(**self.scipy_params)
            testing.assert_allclose(stddev1, stddev2)
        setattr(klass, "test_stddev_cpu", test_stddev_cpu)

        @attr.gpu
        def test_stddev_gpu(self):
            stddev1 = self.gpu_dist.stddev.data
            stddev2 = self.scipy_dist.std(**self.scipy_params)
            testing.assert_allclose(stddev1, stddev2)
        setattr(klass, "test_stddev_gpu", test_stddev_gpu)

        def test_support_cpu(self):
            self.assertEqual(self.cpu_dist.support, self.support)
        setattr(klass, "test_support_cpu", test_support_cpu)

        @attr.gpu
        def test_support_gpu(self):
            self.assertEqual(self.gpu_dist.support, self.support)
        setattr(klass, "test_support_gpu", test_support_gpu)

        def test_survival_cpu(self):
            smp = self.sample_for_test(self.shape)
            survival1 = self.cpu_dist.survival_function(smp).data
            survival2 = self.scipy_dist.sf(smp, **self.scipy_params)
            testing.assert_allclose(survival1, survival2)
        setattr(klass, "test_survival_cpu", test_survival_cpu)

        @attr.gpu
        def test_survival_gpu(self):
            smp = self.sample_for_test(self.shape)
            survival1 = self.gpu_dist.survival_function(cuda.to_gpu(smp)).data
            survival2 = self.scipy_dist.sf(smp, **self.scipy_params)
            testing.assert_allclose(survival1, survival2)
        setattr(klass, "test_survival_gpu", test_survival_gpu)

        def test_variance_cpu(self):
            variance1 = self.cpu_dist.variance.data
            variance2 = self.scipy_dist.var(**self.scipy_params)
            testing.assert_allclose(variance1, variance2)
        setattr(klass, "test_survival_cpu", test_survival_cpu)

        @attr.gpu
        def test_variance_gpu(self):
            variance1 = self.gpu_dist.variance.data
            variance2 = self.scipy_dist.var(**self.scipy_params)
            testing.assert_allclose(variance1, variance2)
        setattr(klass, "test_survival_gpu", test_survival_gpu)

        # Return parameterized class.
        return testing.parameterize(*testing.product({
            'shape': [(3, 2), (1,), ()],
        }))(testing.fix_random()(klass))
    return f
