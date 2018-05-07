import unittest

from chainer import testing
import numpy


def distribution_unittest(dist, scipy_dist, params_init, sample_for_test,
                          support="real", event_shape=(), ignore_tests=set()):
    def f(klass):
        assert issubclass(klass, unittest.TestCase)

        def setUp(self):
            self.dist = dist
            self.scipy_dist = scipy_dist
            self.params, self.scipy_params = params_init(self.shape)
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
            return self.dist(**self.params)
        setattr(klass, "gpu_dist", gpu_dist)

        def test_batch_shape(self):
            self.assertEqual(self.cpu_dist.batch_shape, self.shape)
        setattr(klass, "test_batch_shape", test_batch_shape)

        def test_cdf(self):
            smp = self.sample_for_test(self.shape)
            print(smp, type(smp))
            print(smp + smp, type(smp + smp))
            print(smp - self.params["loc"], type(smp - self.params["loc"]))
            print(self.cpu_dist.loc, type(self.cpu_dist.loc))
            print(smp - self.cpu_dist.loc, type(smp - self.cpu_dist.loc))
            cdf1 = self.cpu_dist.cdf(smp).data
            cdf2 = self.scipy_dist.cdf(smp, **self.scipy_params)
            testing.assert_allclose(cdf1, cdf2)
        setattr(klass, "test_cdf", test_cdf)

        def test_entropy(self):
            ent1 = self.cpu_dist.entropy.data
            ent2 = self.scipy_dist.entropy(**self.scipy_params)
            testing.assert_allclose(ent1, ent2,
                                    atol=1e-2, rtol=1e-2)
        setattr(klass, "test_entropy", test_entropy)

        def test_event_shape(self):
            self.assertEqual(self.cpu_dist.event_shape, self.event_shape)
        setattr(klass, "test_event_shape", test_event_shape)

        def test_icdf(self):
            smp = numpy.random.uniform(
                1e-5, 1 - 1e-5, self.shape).astype(numpy.float32)
            icdf1 = self.cpu_dist.icdf(smp).data
            icdf2 = self.scipy_dist.ppf(smp, **self.scipy_params)
            testing.assert_allclose(icdf1, icdf2)
        setattr(klass, "test_icdf", test_icdf)

        def test_log_cdf(self):
            smp = self.sample_for_test(self.shape)
            log_cdf1 = self.cpu_dist.log_cdf(smp).data
            log_cdf2 = self.scipy_dist.logcdf(smp, **self.scipy_params)
            testing.assert_allclose(log_cdf1, log_cdf2)
        setattr(klass, "test_log_cdf", test_log_cdf)

        def test_log_prob(self):
            smp = self.sample_for_test(self.shape)
            log_prob1 = self.cpu_dist.log_prob(smp).data
            log_prob2 = self.scipy_dist.logpdf(smp, **self.scipy_params)
            testing.assert_allclose(log_prob1, log_prob2)
        setattr(klass, "test_log_prob", test_log_prob)

        def test_log_survival(self):
            smp = self.sample_for_test(self.shape)
            log_survival1 = self.cpu_dist.log_survival_function(smp).data
            log_survival2 = self.scipy_dist.logsf(smp, **self.scipy_params)
            testing.assert_allclose(log_survival1, log_survival2)
        setattr(klass, "test_log_survival", test_log_survival)

        def test_mean(self):
            mean1 = self.cpu_dist.mean.data
            mean2 = self.scipy_dist.mean(**self.scipy_params)
            testing.assert_allclose(mean1, mean2)
        setattr(klass, "test_mean", test_mean)

        def test_mode(self):
            mode1 = self.cpu_dist.mode.data
            mode2 = self.scipy_dist.mean(**self.scipy_params)
            testing.assert_allclose(mode1, mode2)
        setattr(klass, "test_mode", test_mode)

        def test_prob(self):
            smp = self.sample_for_test(self.shape)
            prob1 = self.cpu_dist.prob(smp).data
            prob2 = self.scipy_dist.pdf(smp, **self.scipy_params)
            testing.assert_allclose(prob1, prob2)
        setattr(klass, "test_prob", test_prob)

        def test_sample(self):
            smp1 = self.cpu_dist.sample(shape=(1000000)).data
            smp2 = self.scipy_dist.rvs(**self.scipy_params,
                                       size=(1000000,)+self.shape)
            testing.assert_allclose(smp1.mean(axis=0), smp2.mean(axis=0),
                                    atol=1e-2, rtol=1e-2)
        setattr(klass, "test_sample", test_sample)

        def test_stddev(self):
            stddev1 = self.cpu_dist.stddev.data
            stddev2 = self.scipy_dist.std(**self.scipy_params)
            testing.assert_allclose(stddev1, stddev2)
        setattr(klass, "test_stddev", test_stddev)

        def test_support(self):
            self.assertEqual(self.cpu_dist.support, self.support)
        setattr(klass, "test_support", test_support)

        def test_survival(self):
            smp = self.sample_for_test(self.shape)
            survival1 = self.cpu_dist.survival_function(smp).data
            survival2 = self.scipy_dist.sf(smp, **self.scipy_params)
            testing.assert_allclose(survival1, survival2)
        setattr(klass, "test_survival", test_survival)

        def test_variance(self):
            variance1 = self.cpu_dist.variance.data
            variance2 = self.scipy_dist.var(**self.scipy_params)
            testing.assert_allclose(variance1, variance2)
        setattr(klass, "test_survival", test_survival)

        # Return parameterized class.
        return testing.parameterize(*testing.product({
            'shape': [(3, 2), (1,), ()],
        }))(testing.fix_random()(klass))
    return f
