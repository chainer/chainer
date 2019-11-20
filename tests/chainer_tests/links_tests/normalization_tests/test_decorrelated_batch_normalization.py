import contextlib
import os
import tempfile
import unittest

import numpy

import chainer
from chainer import links
from chainer import testing


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
    'eps': [2e-5, 5e-1],
    'test': [True, False],
    'ndim': [0, 2],
    # NOTE(crcrpar): np.linalg.eigh does not support float16
    'dtype': [numpy.float32, numpy.float64],
})))
@testing.inject_backend_tests(
    None,
    # CPU tests
    [{}]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class DecorrelatedBatchNormalizationTest(testing.LinkTestCase):

    param_names = ()

    def setUp(self):
        self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-3}
        self.check_backward_options = {'atol': 5e-3, 'rtol': 1e-3}
        if self.dtype == numpy.float32:
            self.check_backward_options = {'atol': 5e-2, 'rtol': 5e-2}

    def generate_params(self):
        C = self.n_channels // self.groups
        # TODO(ecastill) mean and projection are not
        # parameters inside the link, just plain arrays
        mean = numpy.random.uniform(
            -1, 1, (self.groups, C)).astype(self.dtype)
        projection = numpy.random.uniform(
            0.5, 1, (self.groups, C, C)).astype(
            self.dtype)
        return mean, projection

    def create_link(self, initializers):
        mean, projection = initializers
        link = links.DecorrelatedBatchNormalization(
            self.n_channels, groups=self.groups, eps=self.eps,
            dtype=self.dtype)
        link.cleargrads()
        if self.test:
            link.avg_mean[...] = mean
            link.avg_projection[...] = projection
        return link

    def generate_inputs(self):
        dtype = self.dtype
        ndim = self.ndim
        shape = (5, self.n_channels) + (2,) * ndim
        m = 5 * 2 ** ndim

        # NOTE(kataoka): The current implementation uses linalg.eigh. Small
        # eigenvalues of the correlation matrix, which can be as small as
        # eps=2e-5, cannot be computed with good *relative* accuracy, but
        # the eigenvalues are used later as `eigvals ** -0.5`. Require the
        # following is sufficiently large:
        # min(eigvals[:k]) == min(singular_vals ** 2 / m + eps)
        min_singular_value = 0.1
        # NOTE(kataoka): Decorrelated batch normalization should be free from
        # "stochastic axis swapping". Requiring a gap between singular values
        # just hides mistakes in implementations.
        min_singular_value_gap = 0.001
        g = self.groups
        zca_shape = g, self.n_channels // g, m
        x = numpy.random.uniform(-1, 1, zca_shape)
        mean = x.mean(axis=2, keepdims=True)
        a = x - mean
        u, s, vh = numpy.linalg.svd(a, full_matrices=False)
        # Decrement the latter dim because of the constraint `sum(_) == 0`
        k = min(zca_shape[1], zca_shape[2] - 1)
        s[:, :k] += (
            min_singular_value
            + min_singular_value_gap * numpy.arange(k)
        )[::-1]
        a = numpy.einsum('bij,bj,bjk->bik', u, s, vh)
        x = a + mean

        x = x.reshape((self.n_channels, shape[0]) + shape[2:]).swapaxes(0, 1)
        x = x.astype(dtype)
        return x,

    def forward_expected(self, link, inputs):
        x, = inputs
        if self.test:
            mean = link.avg_mean
            projection = link.avg_projection
        else:
            mean = _calc_mean(x, self.groups)
            projection = _calc_projection(x, mean,
                                          link.eps, self.groups)
        y_expect = _decorrelated_batch_normalization(
            x, mean, projection, self.groups)
        return y_expect,

    def forward(self, link, inputs, backend_config):
        x, = inputs
        with chainer.using_config('train', not self.test):
            y = link(x)
        return y,


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
