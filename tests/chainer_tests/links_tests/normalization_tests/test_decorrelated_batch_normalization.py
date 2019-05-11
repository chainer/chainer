import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr


def _decorrelated_batch_normalization(x, mean, projection, groups):
    spatial_ndim = len(x.shape[2:])
    spatial_axis = tuple(range(2, 2 + spatial_ndim))
    b, c = x.shape[:2]
    g = groups
    C = c // g

    x = x.reshape((b * g, C) + x.shape[2:])
    x_hat = x.transpose((1, 0) + spatial_axis).reshape((C, -1))

    y_hat = projection.dot(x_hat - mean[:, None])

    y = y_hat.reshape((C, b * g) + x.shape[2:]).transpose(
        (1, 0) + spatial_axis)
    y = y.reshape((-1, c) + x.shape[2:])
    return y


def _calc_projection(x, mean, eps, groups):
    spatial_ndim = len(x.shape[2:])
    spatial_axis = tuple(range(2, 2 + spatial_ndim))
    b, c = x.shape[:2]
    g = groups
    C = c // g
    m = b * g
    for i in spatial_axis:
        m *= x.shape[i]

    x = x.reshape((b * g, C) + x.shape[2:])
    x_hat = x.transpose((1, 0) + spatial_axis).reshape((C, -1))

    mean = x_hat.mean(axis=1)
    x_hat = x_hat - mean[:, None]

    cov = x_hat.dot(x_hat.T) / m + eps * numpy.eye(C, dtype=x.dtype)
    eigvals, eigvectors = numpy.linalg.eigh(cov)
    projection = eigvectors.dot(numpy.diag(eigvals ** -0.5)).dot(eigvectors.T)
    return projection


@testing.parameterize(*(testing.product({
    'n_channels': [8],
    'groups': [1, 2],
    'test': [True, False],
    'ndim': [0, 2],
    # NOTE(crcrpar): np.linalg.eigh does not support float16
    'dtype': [numpy.float32, numpy.float64],
})))
class DecorrelatedBatchNormalizationTest(unittest.TestCase):

    def setUp(self):
        C = self.n_channels // self.groups
        head_ndim = 2

        self.link = links.DecorrelatedBatchNormalization(
            self.n_channels, groups=self.groups, dtype=self.dtype)
        self.link.cleargrads()

        shape = (5, self.n_channels) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)

        if self.test:
            self.mean = numpy.random.uniform(-1, 1, (C,)).astype(self.dtype)
            self.projection = numpy.random.uniform(0.5, 1, (C, C)).astype(
                self.dtype)
            self.link.avg_mean[...] = self.mean
            self.link.avg_projection[...] = self.projection
        else:
            spatial_axis = tuple(range(head_ndim, self.x.ndim))
            x_hat = self.x.reshape((5 * self.groups, C) + self.x.shape[2:])
            x_hat = x_hat.transpose((1, 0) + spatial_axis).reshape((C, -1))
            self.mean = x_hat.mean(axis=1)
            self.projection = _calc_projection(self.x, self.mean,
                                               self.link.eps, self.groups)
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {
            'atol': 5e-3, 'rtol': 1e-3, 'dtype': numpy.float64}
        if self.dtype == numpy.float32:
            self.check_backward_options = {'atol': 5e-2, 'rtol': 5e-2}

    def check_forward(self, x_data):
        with chainer.using_config('train', not self.test):
            x = chainer.Variable(x_data)
            y = self.link(x)
            self.assertEqual(y.dtype, self.dtype)

        y_expect = _decorrelated_batch_normalization(
            self.x, self.mean, self.projection, self.groups)

        testing.assert_allclose(
            y_expect, y.array, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, (),
            eps=1e-2, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
