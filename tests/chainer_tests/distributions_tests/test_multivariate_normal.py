import unittest

import numpy

from chainer import cuda
from chainer import distributions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
    'event_shape': [(3,)],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestMultivariateNormal(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.MultivariateNormal
        self.scipy_dist = stats.multivariate_normal
        self.scipy_onebyone = True
        d, = self.event_shape

        self.test_targets = set([
            'batch_shape', 'entropy', 'event_shape', 'log_prob',
            'support'])

        loc = numpy.random.uniform(
            -1, 1, self.shape + self.event_shape).astype(numpy.float32)
        cov = numpy.random.normal(
            size=(numpy.prod(self.shape),) + (d, d))
        cov = [cov_.dot(cov_.T) for cov_ in cov]
        cov = numpy.vstack(cov).reshape(self.shape + (d, d))
        scale_tril = numpy.linalg.cholesky(cov).astype(numpy.float32)
        self.params = {'loc': loc, 'scale_tril': scale_tril}
        self.scipy_params = {'mean': loc, 'cov': cov}

    def test_value_error(self):
        with self.assertRaises(ValueError):
            self.dist(loc=self.params['loc'])

    def sample_for_test(self):
        smp = numpy.random.normal(
            size=self.sample_shape + self.shape + self.event_shape
        ).astype(numpy.float32)
        return smp


@testing.parameterize(*testing.product({
    'd': [3, 1, 0],
    'lower': [True, False],
    'dtype': [numpy.float32],
}))
@testing.with_requires('scipy')
class TestTriangularInv(unittest.TestCase):

    def setUp(self):
        while True:
            self.x = numpy.random.normal(
                0, 10, size=(self.d, self.d)).astype(self.dtype)
            self.x = numpy.tril(self.x)
            if (abs(self.x.diagonal()) > 0.1).all():
                break
        if not self.lower:
            self.x = self.x.T
        self.gy = numpy.random.normal(size=(self.d, self.d)).astype(self.dtype)
        self.ggy = numpy.random.normal(
            size=(self.d, self.d)).astype(self.dtype)
        self.backward_options = {'atol': 1e-2, 'rtol': 1e-2}
        self.double_backward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def check_forward(self, x_data):
        y = distributions.multivariate_normal._triangular_inv(
            x_data, lower=self.lower)
        inv_x = numpy.linalg.inv(cuda.to_cpu(x_data))
        testing.assert_allclose(y.array, inv_x)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        def f(x):
            return distributions.multivariate_normal._triangular_inv(
                x, lower=self.lower)
        gradient_check.check_backward(
            f, x_data, y_grad, **self.backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad):
        def f(x):
            return distributions.multivariate_normal._triangular_inv(
                x, lower=self.lower)
        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad,
            **self.double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggy)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggy))


@testing.parameterize(*testing.product({
    'd': [3, 5],
    'lower': [True, False],
    'dtype': [numpy.float32],
}))
@testing.without_requires('scipy')
class TestTriangularInvExceptions(unittest.TestCase):
    def setUp(self):
        self.x = numpy.random.normal(
            0, 10, size=(self.d, self.d)).astype(self.dtype)
        self.x = numpy.tril(self.x)
        if not self.lower:
            self.x = self.x.T

    def check_forward(self, x_data):
        with self.assertRaises(ImportError):
            distributions.multivariate_normal._triangular_inv(
                x_data, lower=self.lower)

    def test_forward_cpu(self):
        self.check_forward(self.x)


testing.run_module(__name__, __file__)
