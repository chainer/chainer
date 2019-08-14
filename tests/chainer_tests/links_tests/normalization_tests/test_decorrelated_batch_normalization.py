import contextlib
import os
import tempfile
import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr


def _decorrelated_batch_normalization(x, mean, projection, groups):
    xs = numpy.split(x, groups, axis=1)
    assert mean.shape[0] == groups
    assert projection.shape[0] == groups
    ys = [
        _decorrelated_batch_normalization_1group(xi, m, p)
        for (xi, m, p) in zip(xs, mean, projection)]
    return numpy.concatenate(ys, axis=1)


def _decorrelated_batch_normalization_1group(x, mean, projection):
    spatial_ndim = len(x.shape[2:])
    spatial_axis = tuple(range(2, 2 + spatial_ndim))
    b, C = x.shape[:2]
    x_hat = x.transpose((1, 0) + spatial_axis).reshape(C, -1)
    y_hat = projection.dot(x_hat - mean[:, None])
    y = y_hat.reshape((C, b) + x.shape[2:]).transpose(
        (1, 0) + spatial_axis)
    return y


def _calc_projection(x, mean, eps, groups):
    xs = numpy.split(x, groups, axis=1)
    assert mean.shape[0] == groups
    projections = [
        _calc_projection_1group(xi, m, eps)
        for (xi, m) in zip(xs, mean)]
    return numpy.concatenate([p[None] for p in projections])


def _calc_projection_1group(x, mean, eps):
    spatial_ndim = len(x.shape[2:])
    spatial_axis = tuple(range(2, 2 + spatial_ndim))
    b, C = x.shape[:2]
    m = b
    for i in spatial_axis:
        m *= x.shape[i]

    x_hat = x.transpose((1, 0) + spatial_axis).reshape(C, -1)
    mean = x_hat.mean(axis=1)
    x_hat = x_hat - mean[:, None]
    cov = x_hat.dot(x_hat.T) / m + eps * numpy.eye(C, dtype=x.dtype)
    eigvals, eigvectors = numpy.linalg.eigh(cov)
    projection = eigvectors.dot(numpy.diag(eigvals ** -0.5)).dot(eigvectors.T)
    return projection


def _calc_mean(x, groups):
    axis = (0,) + tuple(range(2, x.ndim))
    return x.mean(axis=axis).reshape(groups, -1)


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

        self.link = links.DecorrelatedBatchNormalization(
            self.n_channels, groups=self.groups, dtype=self.dtype)
        self.link.cleargrads()

        shape = (5, self.n_channels) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)

        if self.test:
            self.mean = numpy.random.uniform(
                -1, 1, (self.groups, C)).astype(self.dtype)
            self.projection = numpy.random.uniform(
                0.5, 1, (self.groups, C, C)).astype(
                self.dtype)
            self.link.avg_mean[...] = self.mean
            self.link.avg_projection[...] = self.projection
        else:
            self.mean = _calc_mean(self.x, self.groups)
            self.projection = _calc_projection(self.x, self.mean,
                                               self.link.eps, self.groups)
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'atol': 5e-3, 'rtol': 1e-3}
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
            dtype=numpy.float64, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


# TODO(kataoka) Use `contextlib.nullcontext` if Python 3.7 or higher is assumed
@contextlib.contextmanager
def nullcontext():
    yield


@testing.parameterize(*(testing.product({
    'n_channels': [8],
    'groups': [1, 2],
    'dtype': [numpy.float32, numpy.float64],
})))
@testing.inject_backend_tests([
    'test_model_compatibility_npz', 'test_model_compatibility_hdf5',
], [
    {},
    {'use_cuda': True},
    {'use_cuda': True, 'cuda_device': 1},
    {'use_chainerx': True, 'chainerx_device': 'native:0'},
    {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
])
class TestDecorrelatedBatchNormalizationCompat(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path

    def tearDown(self):
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def check_model_compatibility(self, backend_config, save, load):
        C = self.n_channels // self.groups
        old_model = {
            'avg_mean': numpy.random.uniform(
                -1, 1, (C,)).astype(self.dtype),
            'avg_projection': numpy.random.uniform(
                0.5, 1, (C, C)).astype(self.dtype),
            'N': numpy.array(0)
        }
        save(self.temp_file_path, old_model)

        model = links.DecorrelatedBatchNormalization(
            self.n_channels, groups=self.groups, dtype=self.dtype)
        model.to_device(backend_config.device)
        with (
                testing.assert_warns(UserWarning) if self.groups != 1
                else nullcontext()):
            load(self.temp_file_path, model)
        x = numpy.random.rand(5, self.n_channels, 2).astype(self.dtype)
        x = backend_config.get_array(x)
        with chainer.using_config('train', False):
            model(x)
        model(x)

    def test_model_compatibility_npz(self, backend_config):
        self.check_model_compatibility(
            backend_config,
            chainer.serializers.save_npz,
            chainer.serializers.load_npz,
        )

    @testing.with_requires('h5py')
    def test_model_compatibility_hdf5(self, backend_config):
        self.check_model_compatibility(
            backend_config,
            chainer.serializers.save_hdf5,
            chainer.serializers.load_hdf5,
        )


testing.run_module(__name__, __file__)
