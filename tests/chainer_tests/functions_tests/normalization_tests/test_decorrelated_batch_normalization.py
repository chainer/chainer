import unittest

import numpy

from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import backend


def _to_fcontiguous(arrays):
    xp = cuda.get_array_module(*arrays)
    return [xp.asfortranarray(a) for a in arrays]


def _decorrelated_batch_normalization(args):
    x, mean, projection, groups = args
    spatial_ndim = len(x.shape[2:])
    spatial_axis = tuple(range(2, 2 + spatial_ndim))
    b, c = x.shape[:2]
    g = groups
    C = c // g

    x = x.reshape((b * g, C, ) + x.shape[2:])
    x_hat = x.transpose((1, 0) + spatial_axis).reshape((C, -1))

    y_hat = projection.dot(x_hat - mean[:, None])

    y = y_hat.reshape((C, b * g,) + x.shape[2:]).transpose(
        (1, 0) + spatial_axis)
    y = y.reshape((-1, c, ) + x.shape[2:])
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

    x = x.reshape((b * g, C, ) + x.shape[2:])
    x_hat = x.transpose((1, 0) + spatial_axis).reshape((C, -1))

    mean = x_hat.mean(axis=1)
    x_hat = x_hat - mean[:, None]

    cov = x_hat.dot(x_hat.T) / m + eps * numpy.eye(C, dtype=x.dtype)
    eigvals, eigvectors = numpy.linalg.eigh(cov)
    projection = eigvectors.dot(numpy.diag(eigvals ** -0.5)).dot(eigvectors.T)
    return projection


@testing.parameterize(*(testing.product({
    'n_channels': [8],
    'ndim': [0, 2],
    'groups': [1, 2],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float32],
    'c_contiguous': [True, False],
}) + testing.product({
    'n_channels': [8],
    'ndim': [1],
    'groups': [1, 2],
    'eps': [2e-5, 5e-1],
    # NOTE(tommi): xp.linalg.eigh does not support float16
    'dtype': [numpy.float32, numpy.float64],
    'c_contiguous': [True, False],
})))
@backend.inject_backend_tests(
    ['test_forward', 'test_backward'],
    # CPU tests
    [{'use_cuda': False}]
    # GPU tests
    + [{'use_cuda': True}]
)
class TestDecorrelatedBatchNormalization(unittest.TestCase):

    def setUp(self):
        C = self.n_channels // self.groups
        dtype = self.dtype
        ndim = self.ndim

        head_ndim = 2
        shape = (5, self.n_channels) + (2,) * ndim
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        gy = numpy.random.uniform(-1, 1, shape).astype(dtype)

        spatial_axis = tuple(range(head_ndim, x.ndim))
        x_hat = x.reshape((5 * self.groups, C, ) + x.shape[2:])
        x_hat = x_hat.transpose((1, 0) + spatial_axis).reshape((C, -1))
        mean = x_hat.mean(axis=1)
        projection = _calc_projection(x, mean, self.eps, self.groups)

        self.decay = 0.9
        self.mean = mean
        self.projection = projection

        self.inputs = [x]
        self.grad_outputs = [gy]

        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
        elif self.dtype == numpy.float32:
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}

    def forward_cpu(self, inputs):
        y_expect = _decorrelated_batch_normalization(
            inputs + [self.mean, self.projection, self.groups])
        return y_expect,

    def check_forward(self, inputs, backend_config):
        y_expected, = self.forward_cpu(inputs)

        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
        if not self.c_contiguous:
            inputs = _to_fcontiguous(inputs)

        with backend_config:
            y = functions.decorrelated_batch_normalization(
                *inputs, groups=self.groups, decay=self.decay,
                eps=self.eps)
        assert y.data.dtype == self.dtype

        testing.assert_allclose(
            y_expected, y.data, **self.check_forward_options)

    def test_forward(self, backend_config):
        self.check_forward(self.inputs, backend_config)

    def check_backward(self, inputs, grad_outputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)
        if not self.c_contiguous:
            inputs = _to_fcontiguous(inputs)
            grad_outputs = _to_fcontiguous(grad_outputs)

        def f(*inputs):
            y = functions.decorrelated_batch_normalization(
                *inputs, groups=self.groups, decay=self.decay,
                eps=self.eps)
            return y,

        with backend_config:
            gradient_check.check_backward(
                f, inputs, grad_outputs,
                **self.check_backward_options)

    def test_backward(self, backend_config):
        self.check_backward(self.inputs, self.grad_outputs, backend_config)


@testing.parameterize(*(testing.product({
    'n_channels': [8],
    'ndim': [0, 1, 2],
    'groups': [1, 2],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float32],
    'c_contiguous': [True, False],
}) + testing.product({
    'n_channels': [8],
    'ndim': [1],
    'groups': [1, 2],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'c_contiguous': [True, False],
})))
@backend.inject_backend_tests(
    ['test_forward'],
    # CPU tests
    [{'use_cuda': False}]
    # GPU tests
    + [{'use_cuda': True}]
)
class TestFixedDecorrelatedBatchNormalization(unittest.TestCase):

    def setUp(self):
        C = self.n_channels // self.groups
        dtype = self.dtype
        ndim = self.ndim

        shape = (5, self.n_channels) + (2,) * ndim
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        mean = numpy.random.uniform(-1, 1, C).astype(dtype)
        projection = numpy.random.uniform(0.5, 1, (C, C)).astype(dtype)

        self.inputs = [x, mean, projection]

        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def forward_cpu(self, inputs):
        y_expect = _decorrelated_batch_normalization(inputs + [self.groups, ])
        return y_expect,

    def check_forward(self, inputs, backend_config):
        y_expected, = self.forward_cpu(inputs)

        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
        if not self.c_contiguous:
            inputs = _to_fcontiguous(inputs)

        with backend_config:
            y = functions.fixed_decorrelated_batch_normalization(
                *inputs, groups=self.groups)
        assert y.data.dtype == self.dtype

        testing.assert_allclose(
            y_expected, y.data, **self.check_forward_options)

    def test_forward(self, backend_config):
        self.check_forward(self.inputs, backend_config)


testing.run_module(__name__, __file__)
