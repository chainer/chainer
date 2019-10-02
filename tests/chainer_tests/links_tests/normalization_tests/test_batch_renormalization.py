import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import links
from chainer import testing
from chainer.testing import attr


def _batch_renormalization(expander, gamma, beta, x, mean, std, test,
                           r, d):
    mean = mean[expander]
    std = std[expander]
    if test:
        r, d = 1, 0
    y_expect = gamma * ((x - mean) / std * r + d) + beta
    return y_expect


@testing.parameterize(*(testing.product({
    'test': [True, False],
    'ndim': [0, 1, 2, 3],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'eps': [2e-5, 0.5],
})))
class BatchRenormalizationTest(unittest.TestCase):

    def setUp(self):
        self.expander = (None, Ellipsis) + (None,) * self.ndim
        self.aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))

        self.rmax = self.dtype(3)
        self.dmax = self.dtype(5)

        self.link = links.BatchRenormalization(
            3, rmax=self.rmax, dmax=self.dmax,
            dtype=self.dtype, eps=self.eps)
        gamma = self.link.gamma.array
        gamma[...] = numpy.random.uniform(.5, 1, gamma.shape)
        beta = self.link.beta.array
        beta[...] = numpy.random.uniform(-1, 1, beta.shape)
        self.link.cleargrads()

        self.gamma = gamma.copy()[self.expander]  # fixed on CPU
        self.beta = beta.copy()[self.expander]   # fixed on CPU

        shape = (5, 3) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)

        if self.test:
            self.mean = numpy.random.uniform(-1, 1, (3,)).astype(self.dtype)
            self.var = numpy.random.uniform(0.5, 1, (3,)).astype(self.dtype)
            self.running_mean = self.mean
            self.running_var = self.var
        else:
            self.mean = self.x.mean(axis=self.aggr_axes)
            self.var = self.x.var(axis=self.aggr_axes)
            # Need to add some noise to running_mean and running_var,
            # otherwise we will always get r=1, d=0
            # Note that numpy.exp(3) > rmax ** 2 and 7 > dmax
            self.running_var = self.var * numpy.exp(
                numpy.random.uniform(-3, 3, self.var.shape)).astype(self.dtype)
            self.running_mean = self.mean + (
                (numpy.sqrt(self.running_var) + 0.1)
                * numpy.random.uniform(-7, 7, self.mean.shape)
            ).astype(self.dtype)
        self.link.avg_mean[...] = self.running_mean
        self.link.avg_var[...] = self.running_var
        self.check_forward_optionss = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_optionss = {'atol': 1e-4, 'rtol': 1e-3}
        if self.dtype == numpy.float16:
            self.check_forward_optionss = {'atol': 1e-2, 'rtol': 5e-3}
            self.check_backward_optionss = {'atol': 5e-1, 'rtol': 1e-1}

    def check_forward(self, x_data):
        with chainer.using_config('train', not self.test):
            x = chainer.Variable(x_data)
            y = self.link(x)
            self.assertEqual(y.dtype, self.dtype)

        sigma_batch = numpy.sqrt(self.var + self.eps)
        running_sigma = numpy.sqrt(self.running_var + self.eps)
        r = numpy.clip(sigma_batch / running_sigma, 1.0 / self.rmax, self.rmax)
        d = numpy.clip((self.mean - self.running_mean) / running_sigma,
                       -self.dmax, self.dmax)
        y_expect = _batch_renormalization(
            self.expander, self.gamma, self.beta, self.x, self.mean,
            sigma_batch, self.test,
            r[self.expander], d[self.expander])

        testing.assert_allclose(
            y.array, y_expect, **self.check_forward_optionss)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    @attr.multi_gpu(2)
    def test_forward_multi_gpu(self):
        with cuda.get_device_from_id(1):
            with testing.assert_warns(DeprecationWarning):
                self.link.to_gpu()
            x = cuda.to_gpu(self.x)
        with cuda.get_device_from_id(0):
            self.check_forward(x)


@testing.parameterize(
    {'nx': 10, 'ny': 10, 'eps': 2e-5},
    {'nx': 10, 'ny': 10, 'eps': 1e-1},
    # TODO(Kenta Oono)
    # Pass the case below (this test does not pass when nx != ny).
    # {'nx': 10, 'ny': 15}
)
class TestPopulationStatistics(unittest.TestCase):

    def setUp(self):
        self.decay = 0.9
        self.size = 3
        self.link = links.BatchRenormalization(
            self.size, decay=self.decay, eps=self.eps)
        self.x = numpy.random.uniform(
            -1, 1, (self.nx, self.size)).astype(numpy.float32)
        self.y = numpy.random.uniform(
            -1, 1, (self.ny, self.size)).astype(numpy.float32)

    def check_statistics(self, x, y):
        x = chainer.Variable(x)
        self.link(x, finetune=True)
        mean = self.x.mean(axis=0)
        testing.assert_allclose(self.link.avg_mean, mean)
        unbiased_var = self.x.var(axis=0) * self.nx / (self.nx - 1)
        testing.assert_allclose(self.link.avg_var, unbiased_var)

        y = chainer.Variable(y)
        with chainer.using_config('train', False):
            self.link(y, finetune=True)
        testing.assert_allclose(self.link.avg_mean, mean)
        testing.assert_allclose(self.link.avg_var, unbiased_var)

    def test_statistics_cpu(self):
        self.check_statistics(self.x, self.y)

    @attr.gpu
    def test_statistics_gpu(self):
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        self.check_statistics(cuda.to_gpu(self.x), cuda.to_gpu(self.y))

    def check_statistics2(self, x, y):
        x = chainer.Variable(x)
        y = chainer.Variable(y)
        self.link(x, finetune=True)
        self.link(y, finetune=True)
        mean = (self.x.sum(axis=0) + self.y.sum(axis=0)) / (self.nx + self.ny)
        var = (self.x.var(axis=0) * self.nx +
               self.y.var(axis=0) * self.ny) / (self.nx + self.ny)

        # TODO(Kenta Oono)
        # Fix the estimate of the unbiased variance.
        # Unbiased variance should be (nx + ny) / (nx + ny - 1) times of
        # the variance.
        # But the multiplier is ny / (ny - 1) in current implementation
        # these two values are different when nx is not equal to ny.
        unbiased_var = var * self.ny / (self.ny - 1)
        testing.assert_allclose(self.link.avg_mean, mean)
        testing.assert_allclose(self.link.avg_var, unbiased_var)

    def test_statistics2_cpu(self):
        self.check_statistics2(self.x, self.y)

    @attr.gpu
    def test_statistics2_gpu(self):
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        self.check_statistics2(
            cuda.to_gpu(self.x),
            cuda.to_gpu(self.y))


testing.run_module(__name__, __file__)
