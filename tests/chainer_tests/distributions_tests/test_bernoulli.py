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
    'extreme_values': [True, False],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestBernoulli(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Bernoulli
        self.scipy_dist = stats.bernoulli

        self.test_targets = set([
            "batch_shape", "entropy", "log_prob", "mean", "prob", "sample",
            "stddev", "support", "variance"])

        if self.extreme_values:
            p = numpy.random.randint(0, 2, self.shape).astype(numpy.float32)
        else:
            p = numpy.random.uniform(0, 1, self.shape).astype(numpy.float32)

        self.params = {"p": p}
        self.scipy_params = {"p": p}

        self.support = '{0, 1}'
        self.continuous = False

        self.old_settings = None
        if self.extreme_values:
            self.old_settings = numpy.seterr(divide='ignore', invalid='ignore')

    def tearDown(self):
        if self.old_settings is not None:
            numpy.seterr(**self.old_settings)

    def sample_for_test(self):
        smp = numpy.random.randint(
            2, size=self.sample_shape + self.shape).astype(numpy.float32)
        return smp


@testing.parameterize(*testing.product({
    'shape': [(2, 3), ()],
    'dtype': [numpy.float32, numpy.float64],
}))
class TestBernoulliLogProb(unittest.TestCase):

    def setUp(self):
        self.logit = numpy.random.normal(size=self.shape).astype(self.dtype)
        self.x = numpy.random.randint(0, 2, size=self.shape).astype(self.dtype)
        self.gy = numpy.random.normal(size=self.shape).astype(self.dtype)
        self.ggx = numpy.random.normal(size=self.shape).astype(self.dtype)
        self.backward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def check_forward(self, x_data, logit_data):
        distributions.bernoulli._bernoulli_log_prob(x_data, logit_data)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.logit)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.logit))

    def check_backward(self, logit_data, x_data, y_grad):
        def f(logit):
            return distributions.bernoulli._bernoulli_log_prob(
                logit, x_data)
        gradient_check.check_backward(
            f, logit_data, y_grad, **self.backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.logit, self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.logit), cuda.to_gpu(self.x),
                            cuda.to_gpu(self.gy))

    def check_double_backward(self, logit_data, x_data, y_grad, x_grad_grad):
        def f(logit):
            return distributions.bernoulli._bernoulli_log_prob(
                logit, x_data)
        gradient_check.check_double_backward(
            f, logit_data, y_grad, x_grad_grad, dtype=numpy.float64,
            **self.backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.logit, self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.logit), cuda.to_gpu(self.x),
            cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))


testing.run_module(__name__, __file__)
