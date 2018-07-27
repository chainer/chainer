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
class TestBeta(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Beta
        self.scipy_dist = stats.beta

        self.test_targets = set([
            "batch_shape", "entropy", "event_shape", "log_prob", "mean",
            "sample", "support", "variance"])

        a = numpy.random.uniform(0, 10, self.shape).astype(numpy.float32)
        b = numpy.random.uniform(0, 10, self.shape).astype(numpy.float32)
        self.params = {"a": a, "b": b}
        self.scipy_params = {"a": a, "b": b}

        self.support = "[0, 1]"

    def sample_for_test(self):
        smp = numpy.random.uniform(
            size=self.sample_shape + self.shape).astype(numpy.float32)
        return smp


@testing.parameterize(*testing.product({
    'shape': [(2, 3), ()],
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestLBeta(unittest.TestCase):

    def setUp(self):
        self.a = numpy.random.uniform(
            low=0.1, high=0.9, size=self.shape).astype(self.dtype)
        self.b = numpy.random.uniform(
            low=0.1, high=0.9, size=self.shape).astype(self.dtype)
        self.gy = numpy.random.normal(size=self.shape).astype(self.dtype)
        self.gga = numpy.random.normal(size=self.shape).astype(self.dtype)
        self.ggb = numpy.random.normal(size=self.shape).astype(self.dtype)
        self.backward_options = {'atol': 1e-2, 'rtol': 1e-2, 'eps': 1e-4}

    def check_forward(self, a_data, b_data):
        y = distributions.beta._lbeta(a_data, b_data)
        import scipy.special
        y2 = scipy.special.betaln(cuda.to_cpu(a_data), cuda.to_cpu(b_data))
        testing.assert_allclose(y.data, y2)

    def test_forward_cpu(self):
        self.check_forward(self.a, self.b)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.a), cuda.to_gpu(self.b))

    def check_backward(self, a_data, b_data, y_grad):
        gradient_check.check_backward(
            distributions.beta._lbeta,
            (a_data, b_data), y_grad, **self.backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.a, self.b, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.a), cuda.to_gpu(self.b),
                            cuda.to_gpu(self.gy))

    def check_double_backward(self, a_data, b_data, y_grad, a_grad_grad,
                              b_grad_grad):
        gradient_check.check_double_backward(
            distributions.beta._lbeta, (a_data, b_data), y_grad,
            (a_grad_grad, b_grad_grad),
            dtype=self.dtype, **self.backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.a, self.b, self.gy, self.gga, self.ggb)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.a), cuda.to_gpu(self.b),
                                   cuda.to_gpu(self.gy), cuda.to_gpu(self.gga),
                                   cuda.to_gpu(self.ggb))


testing.run_module(__name__, __file__)
