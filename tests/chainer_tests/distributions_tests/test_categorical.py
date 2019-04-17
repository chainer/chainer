import numpy

from chainer import cuda
from chainer import distributions
from chainer import testing
from chainer.testing import array


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
    # 'extreme_values': [True, False],
    'extreme_values': [False],
    'logit_option': [True, False]
}))
@testing.fix_random()
@testing.with_requires('scipy>=0.19.0')
class TestCategorical(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Categorical
        self.scipy_dist = stats.multinomial

        self.test_targets = set([
            'batch_shape', 'event_shape', 'entropy', 'log_prob', 'sample'])

        if self.logit_option:
            if self.extreme_values:
                logit = -numpy.inf \
                    * numpy.ones((3,)+self.shape).astype(numpy.float32)
                logit[0] = 0.
                logit = numpy.rollaxis(logit, 0, logit.ndim)
            else:
                logit = numpy.random.normal(
                    size=self.shape+(3,)).astype(numpy.float32)
            p = numpy.exp(logit)
            p /= numpy.expand_dims(p.sum(axis=-1), axis=-1)
            self.params = {'logit': logit}
        else:
            if self.extreme_values:
                p = numpy.zeros((3,)+self.shape).astype(numpy.float32)
                p[0] = 1.
                p = numpy.rollaxis(p, 0, p.ndim)
            else:
                logit = numpy.random.normal(
                    size=self.shape+(3,)).astype(numpy.float32)
                p = numpy.exp(logit)
                p /= numpy.expand_dims(p.sum(axis=-1), axis=-1)
            self.params = {'p': p}

        n = numpy.ones(self.shape)
        self.scipy_params = {'n': n, 'p': p}

        self.continuous = False

        self.old_settings = None
        if self.extreme_values:
            self.old_settings = numpy.seterr(divide='ignore', invalid='ignore')

    def tearDown(self):
        if self.old_settings is not None:
            numpy.seterr(**self.old_settings)

    def sample_for_test(self):
        smp = numpy.random.randint(
            0, 3, self.sample_shape + self.shape).astype(numpy.int32)
        return smp

    def check_log_prob(self, is_gpu):
        smp = self.sample_for_test()
        if is_gpu:
            log_prob1 = self.gpu_dist.log_prob(cuda.to_gpu(smp)).data
        else:
            log_prob1 = self.cpu_dist.log_prob(smp).data

        scipy_prob = self.scipy_dist.logpmf

        onebyone_smp = smp.reshape(
            (int(numpy.prod(self.sample_shape)),
             numpy.prod(self.shape),
             int(numpy.prod(self.event_shape))))
        onebyone_smp = numpy.swapaxes(onebyone_smp, 0, 1)
        onebyone_smp = onebyone_smp.reshape((-1,) + self.sample_shape
                                            + self.event_shape)

        log_prob2 = []
        for one_params, one_smp in zip(
                self.scipy_onebyone_params_iter(), onebyone_smp):
            one_smp = numpy.eye(3)[one_smp]
            log_prob2.append(scipy_prob(one_smp, **one_params))
        log_prob2 = numpy.vstack(log_prob2)
        log_prob2 = log_prob2.reshape(numpy.prod(self.shape), -1).T
        log_prob2 = log_prob2.reshape(self.sample_shape + self.shape)
        array.assert_allclose(log_prob1, log_prob2)

    def check_sample(self, is_gpu):
        if is_gpu:
            smp1 = self.gpu_dist.sample(
                sample_shape=(100000,)+self.sample_shape).data
        else:
            smp1 = self.cpu_dist.sample(
                sample_shape=(100000,)+self.sample_shape).data

        smp2 = []
        for one_params in self.scipy_onebyone_params_iter():
            smp2.append(self.scipy_dist.rvs(
                size=(100000,)+self.sample_shape, **one_params))
        smp2 = numpy.vstack(smp2)
        smp2 = smp2.dot(numpy.arange(3))
        smp2 = smp2.reshape((numpy.prod(self.shape), 100000)
                            + self.sample_shape
                            + self.cpu_dist.event_shape)
        smp2 = numpy.rollaxis(
            smp2, 0, smp2.ndim-len(self.cpu_dist.event_shape))
        smp2 = smp2.reshape((100000,) + self.sample_shape + self.shape
                            + self.cpu_dist.event_shape)
        array.assert_allclose(smp1.mean(axis=0), smp2.mean(axis=0),
                              atol=3e-2, rtol=3e-2)
        array.assert_allclose(smp1.std(axis=0), smp2.std(axis=0),
                              atol=3e-2, rtol=3e-2)


testing.run_module(__name__, __file__)
