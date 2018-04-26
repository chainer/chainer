import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _decorrelated_batch_normalization(args):
    x, mean, projection = args
    spatial_ndim = len(x.shape[2:])
    spatial_axis = tuple(range(2, 2 + spatial_ndim))
    b, c = x.shape[:2]

    x_hat = x.transpose((1, 0) + spatial_axis).reshape((c, -1))

    y_hat = projection.dot(x_hat - mean[:, None])

    y = y_hat.reshape((c, b,) + x.shape[2:]).transpose((1, 0) + spatial_axis)
    return y


def _calc_projection(x, mean, eps):
    spatial_ndim = len(x.shape[2:])
    spatial_axis = tuple(range(2, 2 + spatial_ndim))
    b, c = x.shape[:2]
    m = b * numpy.prod(numpy.array([x.shape[i] for i in spatial_axis]))

    x_hat = x.transpose((1, 0) + spatial_axis).reshape((c, -1))

    mean = x_hat.mean(axis=1)
    x_hat = x_hat - mean[:, None]

    cov = x_hat.dot(x_hat.T) / m + eps * numpy.eye(c, dtype=x.dtype)
    eigvals, eigvectors = numpy.linalg.eigh(cov)
    projection = eigvectors.dot(numpy.diag(eigvals ** -0.5)).dot(eigvectors.T)
    return projection


@testing.parameterize(*(testing.product({
    'test': [True, False],
    'ndim': [0, 2],
    # NOTE(tommi): xp.linalg.eigh does not support float16
    'dtype': [numpy.float32, numpy.float64],
})))
class BatchNormalizationTest(unittest.TestCase):

    def setUp(self):
        self.aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))

        self.link = links.DecorrelatedBatchNormalization(3, dtype=self.dtype)
        self.link.cleargrads()

        shape = (5, 3) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)

        if self.test:
            self.mean = numpy.random.uniform(-1, 1, (3,)).astype(self.dtype)
            self.projection = numpy.random.uniform(0.5, 1, (3, 3)).astype(
                self.dtype)
            self.link.expected_mean[...] = self.mean
            self.link.expected_projection[...] = self.projection
        else:
            self.mean = self.x.mean(axis=self.aggr_axes)
            self.projection = _calc_projection(self.x, self.mean,
                                               self.link.eps)
        self.check_forward_optionss = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_optionss = {'atol': 1e-4, 'rtol': 1e-3}
        if self.dtype == numpy.float16:
            self.check_forward_optionss = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_optionss = {'atol': 5e-1, 'rtol': 1e-1}

    def check_forward(self, x_data):
        with chainer.using_config('train', not self.test):
            x = chainer.Variable(x_data)
            y = self.link(x)
            self.assertEqual(y.data.dtype, self.dtype)

        y_expect = _decorrelated_batch_normalization((
            self.x, self.mean, self.projection))

        testing.assert_allclose(
            y_expect, y.data, **self.check_forward_optionss)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, (),
            eps=1e-2, **self.check_backward_optionss)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
