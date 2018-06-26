import unittest

from chainer.backends import cuda
from chainer import distributions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
import numpy


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestLaplace(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Laplace
        self.scipy_dist = stats.laplace

        self.test_targets = set([
            "batch_shape", "cdf", "entropy", "event_shape", "log_prob",
            "mean", "prob", "sample", "stddev", "support", "variance"])

        loc = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        scale = numpy.exp(
            numpy.random.uniform(-1, 1, self.shape)).astype(numpy.float32)
        self.params = {"loc": loc, "scale": scale}
        self.scipy_params = {"loc": loc, "scale": scale}

    def sample_for_test(self):
        smp = numpy.random.normal(
            size=self.sample_shape + self.shape).astype(numpy.float32)
        return smp


@testing.parameterize(*testing.product({
    'shape': [(2, 3), ()],
    'dtype': [numpy.float32, numpy.float64],
}))
class TestLaplaceCDF(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.normal(size=self.shape).astype(self.dtype)
        self.gy = numpy.random.normal(size=self.shape).astype(self.dtype)
        self.backward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def forward(self, x):
        y, = distributions.laplace.LaplaceCDF().apply((x,))
        return y

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.forward, x_data, y_grad, **self.backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x1), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
