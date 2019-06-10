import numpy

from chainer import functions
from chainer import testing


def _decorrelated_batch_normalization(x, mean, projection, groups):
    spatial_ndim = len(x.shape[2:])
    spatial_axis = tuple(range(2, 2 + spatial_ndim))
    b, c = x.shape[:2]
    g = groups
    C = c // g
    x = x.reshape((b * g, C) + x.shape[2:])
    x_hat = x.transpose((1, 0) + spatial_axis).reshape(C, -1)
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
    x_hat = x.transpose((1, 0) + spatial_axis).reshape(C, -1)
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
        self.decay = 0.9
        check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        check_backward_options = {'atol': 1e-4, 'rtol': 1e-3}
        if self.dtype == numpy.float16:
            check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
        elif self.dtype == numpy.float32:
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
        return functions.decorrelated_batch_normalization(
            x, groups=self.groups, eps=self.eps),

    def forward_expected(self, inputs):
        x, = inputs
        C = self.n_channels // self.groups
        head_ndim = 2
        spatial_axis = tuple(range(head_ndim, x.ndim))
        x_hat = x.reshape((5 * self.groups, C) + x.shape[2:])
        x_hat = x_hat.transpose((1, 0) + spatial_axis).reshape(C, -1)
        mean = x_hat.mean(axis=1)
        projection = _calc_projection(x, mean, self.eps, self.groups)

        return _decorrelated_batch_normalization(
            x, mean, projection, self.groups),


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
        self.mean = numpy.random.uniform(-1, 1, C).astype(dtype)
        self.projection = numpy.random.uniform(0.5, 1, (C, C)).astype(dtype)

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
