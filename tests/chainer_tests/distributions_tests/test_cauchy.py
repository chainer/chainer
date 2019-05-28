import numpy

from chainer.backends import cuda
from chainer import distributions
from chainer import testing
from chainer.testing import array
from chainer.testing import attr
from chainer import utils


@testing.parameterize(*testing.product({
    'shape': [(2, 3), ()],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestCauchy(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Cauchy
        self.scipy_dist = stats.cauchy

        self.test_targets = set(['batch_shape', 'cdf', 'entropy',
                                 'event_shape', 'icdf', 'log_prob',
                                 'support'])

        loc = utils.force_array(
            numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32))
        scale = utils.force_array(numpy.exp(
            numpy.random.uniform(-1, 1, self.shape)).astype(numpy.float32))
        self.params = {'loc': loc, 'scale': scale}
        self.scipy_params = {'loc': loc, 'scale': scale}

    def sample_for_test(self):
        smp = numpy.random.normal(
            size=self.sample_shape + self.shape).astype(numpy.float32)
        return smp

    def check_mean(self, is_gpu):
        with testing.assert_warns(RuntimeWarning):
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

    def test_mean_cpu(self):
        self.check_mean(False)

    @attr.gpu
    def test_mean_gpu(self):
        self.check_mean(True)

    def check_sample(self, is_gpu):
        if is_gpu:
            smp1 = self.gpu_dist.sample(
                sample_shape=(100000,)+self.sample_shape).data
            smp1 = cuda.to_cpu(smp1)
        else:
            smp1 = self.cpu_dist.sample(
                sample_shape=(100000,)+self.sample_shape).data
        smp2 = self.scipy_dist.rvs(
            size=(100000,)+self.sample_shape+self.shape, **self.scipy_params)
        testing.assert_allclose(numpy.median(smp1, axis=0),
                                numpy.median(smp2, axis=0),
                                atol=3e-2, rtol=3e-2)

    def test_sample_cpu(self):
        self.check_sample(False)

    @attr.gpu
    def test_sample_gpu(self):
        self.check_sample(True)

    def check_variance(self, is_gpu):
        with testing.assert_warns(RuntimeWarning):
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

    def test_variance_cpu(self):
        self.check_variance(False)

    @attr.gpu
    def test_variance_gpu(self):
        self.check_variance(True)


testing.run_module(__name__, __file__)
