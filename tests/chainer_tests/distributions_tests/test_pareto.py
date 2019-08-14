import numpy

from chainer import cuda
from chainer import distributions
from chainer import testing


@testing.parameterize(*testing.product({
    'shape': [(2, 3), ()],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestPareto(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Pareto
        self.scipy_dist = stats.pareto

        self.test_targets = set([
            'batch_shape', 'entropy', 'event_shape', 'log_prob',
            'mean', 'support', 'variance'])

        scale = numpy.exp(numpy.random.uniform(
            -1, 1, self.shape)).astype(numpy.float32)
        alpha = numpy.exp(numpy.random.uniform(
            -1, 1, self.shape)).astype(numpy.float32)
        scale, alpha = numpy.asarray(scale), numpy.asarray(alpha)
        self.params = {'scale': scale, 'alpha': alpha}
        self.scipy_params = {'scale': scale, 'b': alpha}

        self.support = '[scale, inf]'

    def sample_for_test(self):
        smp = numpy.random.pareto(
            a=1, size=self.sample_shape + self.shape).astype(numpy.float32)
        return smp

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

    @testing.attr.gpu
    def test_sample_gpu(self):
        self.check_sample(True)


testing.run_module(__name__, __file__)
