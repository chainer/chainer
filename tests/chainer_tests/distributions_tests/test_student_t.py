import numpy

from chainer import distributions
from chainer import testing


@testing.parameterize(*testing.product({
    'shape': [(2, 3), ()],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestStudentT(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.StudentT
        self.scipy_dist = stats.t

        self.test_targets = set(
            ["batch_shape", "entropy", "event_shape", "log_prob", "mean",
             "support", "variance"])

        nu = numpy.random.uniform(1, 10, self.shape).astype(numpy.float32)
        loc = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        scale = numpy.exp(
            numpy.random.uniform(-1, 1, self.shape)).astype(numpy.float32)
        scale = numpy.asarray(scale)
        self.params = {"nu": nu, "loc": loc, "scale": scale}
        self.scipy_params = {"df": nu, "loc": loc, "scale": scale}

    def sample_for_test(self):
        smp = numpy.random.normal(
            size=self.sample_shape + self.shape).astype(numpy.float32)
        return smp

    def check_sample(self, is_gpu):
        smp1, smp2 = self.sample_from_chainer_and_scipy(is_gpu)
        if is_gpu:
            smp1 = smp1.get()

        testing.assert_allclose(numpy.median(smp1, axis=0),
                                numpy.median(smp2, axis=0),
                                atol=3e-2, rtol=3e-2)

    def test_sample_cpu(self):
        self.check_sample(False)

    @testing.attr.gpu
    def test_sample_gpu(self):
        self.check_sample(True)


testing.run_module(__name__, __file__)
