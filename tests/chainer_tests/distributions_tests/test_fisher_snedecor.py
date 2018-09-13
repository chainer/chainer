from chainer.backends import cuda
from chainer import distributions
from chainer import testing
from chainer.testing import attr
import numpy


@testing.parameterize(*testing.product({
    'shape': [(2, 3), ()],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestFisherSnedecor(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.FisherSnedecor
        self.scipy_dist = stats.f

        self.test_targets = set([
            "batch_shape", "event_shape", "log_prob", "mean", "support",
            "variance"])

        d1 = numpy.random.uniform(1, 7, self.shape).astype(numpy.float32)
        d2 = numpy.random.uniform(1, 7, self.shape).astype(numpy.float32)
        self.params = {"d1": d1, "d2": d2}
        self.scipy_params = {"dfn": d1, "dfd": d2}

        self.support = 'positive'

    def sample_for_test(self):
        smp = numpy.exp(numpy.random.normal(
            size=self.sample_shape + self.shape)).astype(numpy.float32)
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

    @attr.gpu
    def test_sample_gpu(self):
        self.check_sample(True)


testing.run_module(__name__, __file__)
