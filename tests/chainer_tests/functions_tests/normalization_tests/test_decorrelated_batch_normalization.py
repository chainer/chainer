import numpy

from chainer import functions
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
    'ndim': [0, 2],
    'groups': [1, 2],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float32],
    'contiguous': ['C', None],
}) + testing.product({
    'n_channels': [8],
    'ndim': [1],
    'groups': [1, 2],
    'eps': [2e-5, 5e-1],
    # NOTE(crcrpar): np.linalg.eigh does not support float16
    'dtype': [numpy.float32, numpy.float64],
    'contiguous': ['C', None],
})))
@testing.backend.inject_backend_tests(
    None,
    # CPU tests
    [{}]
    # GPU tests
    + [{'use_cuda': True}]
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0']
    })
)
class TestDecorrelatedBatchNormalization(testing.FunctionTestCase):

    # TODO(crcrpar): Delete this line once double backward of
    # :func:`~chainer.functions.decorrelated_batch_normalization` is
    # implemented.
    skip_double_backward_test = True

    def setUp(self):
        check_forward_options = {'atol': 1e-3, 'rtol': 1e-3}
        check_backward_options = {'atol': 1e-3, 'rtol': 1e-3}
        if self.dtype == numpy.float32:
            check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
        self.check_forward_options = check_forward_options
        self.check_backward_options = check_backward_options

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

    def forward(self, inputs, device):
        x, = inputs
        return functions.decorrelated_batch_normalization(
            x, groups=self.groups, eps=self.eps),

    def forward_expected(self, inputs):
        x, = inputs
        groups = self.groups
        mean = _calc_mean(x, groups)
        projection = _calc_projection(x, mean, self.eps, groups)
        return _decorrelated_batch_normalization(
            x, mean, projection, groups),


@testing.parameterize(*(testing.product({
    'n_channels': [8],
    'ndim': [0, 1, 2],
    'groups': [1, 2],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float32],
    'contiguous': ['C', None],
}) + testing.product({
    'n_channels': [8],
    'ndim': [1],
    'groups': [1, 2],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'contiguous': ['C', None],
})))
@testing.backend.inject_backend_tests(
    None,
    # CPU tests
    [{}]
    # GPU tests
    + [{'use_cuda': True}]
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
)
class TestFixedDecorrelatedBatchNormalization(testing.FunctionTestCase):

    # TODO(crcrpar): Delete this line once double backward of
    # :func:`~chainer.functions.fixed_decorrelated_batch_normalization` is
    # implemented.
    skip_double_backward_test = True

    def setUp(self):
        C = self.n_channels // self.groups
        dtype = self.dtype
        self.mean = numpy.random.uniform(
            -1, 1, (self.groups, C)).astype(dtype)
        self.projection = numpy.random.uniform(
            0.5, 1, (self.groups, C, C)).astype(dtype)

        check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        check_backward_options = {'atol': 1e-4, 'rtol': 1e-3}
        if self.dtype == numpy.float16:
            check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
        if self.dtype == numpy.float32:
            check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
        self.check_forward_options = check_forward_options
        self.check_backward_options = check_backward_options

    def generate_inputs(self):
        dtype = self.dtype
        ndim = self.ndim
        shape = (5, self.n_channels) + (2,) * ndim
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        mean = device.send_array(self.mean.copy())
        projection = device.send_array(self.projection.copy())
        return functions.fixed_decorrelated_batch_normalization(
            x, mean, projection, groups=self.groups
        ),

    def forward_expected(self, inputs):
        x, = inputs
        mean = self.mean.copy()
        projection = self.projection.copy()
        return _decorrelated_batch_normalization(
            x, mean, projection, self.groups),


testing.run_module(__name__, __file__)
