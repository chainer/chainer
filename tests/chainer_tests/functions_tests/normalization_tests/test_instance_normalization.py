import numpy

from chainer import functions
from chainer import testing
from chainer.testing import backend


def _instance_normalization(x, gamma, beta, running_mean=None,
                            running_var=None, eps=2e-5, decay=0.9):
    org_shape = x.shape
    b, c = org_shape[:2]
    x_reshaped = x.reshape((1, b * c) + org_shape[2:])
    aggr_axes = (0,) + tuple(range(2, len(org_shape)))
    expander = [Ellipsis] * len(org_shape)
    for i in aggr_axes:
        expander[i] = None
    expander = tuple(expander)
    mean = numpy.mean(x_reshaped, axis=aggr_axes)
    var = numpy.var(x_reshaped, axis=aggr_axes)
    std = numpy.sqrt(var + eps)
    x_reshaped_normalized = (x_reshaped - mean[expander]) / std[expander]
    x_normalized = x_reshaped_normalized.reshape(org_shape)
    y = gamma[expander] * x_normalized + beta[expander]
    return y


def _fixed_instance_normalization(
        x, gamma, beta, mean, var, eps=2e-5, decay=0.9):
    org_shape = x.shape
    aggr_axes = (0,) + tuple(range(2, len(org_shape)))
    expander = [Ellipsis] * len(org_shape)
    for i in aggr_axes:
        expander[i] = None
    expander = tuple(expander)
    x_normalized = (x - mean[expander]) / numpy.sqrt(var[expander] + eps)
    y = gamma[expander] * x_normalized + beta[expander]
    return y


@testing.parameterize(*(testing.product({
    'shape': [(1, 4, 5, 5), (5, 4, 15)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'running_statistics': [True, False],
    'eps': [2e-5],
    'decay': [0.9],
})))
@backend.inject_backend_tests(
    None,
    # # CPU tests
    testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
        'cudnn_fast_batch_normalization': [True, False],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
)
class TestInstanceNormalization(testing.FunctionTestCase):

    def setUp(self):
        self.check_forward_options.update({'atol': 1e-4, 'rtol': 1e-3})
        self.check_backward_options.update({'atol': 1e-5, 'rtol': 1e-4})
        self.check_double_backward_options.update({'atol': 1e-3, 'rtol': 1e-2})
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 1e-2, 'rtol': 1e-2})
            self.check_backward_options.update({'atol': 1e-2, 'rtol': 1e-2})
            self.check_double_backward_options.update(
                {'atol': 1e-2, 'rtol': 1e-2})
        if self.running_statistics:
            c = self.shape[1]
            self.mean = numpy.random.uniform(-1, 1, c).astype(self.dtype)
            self.var = numpy.random.uniform(-1, 1, c).astype(self.dtype)

    def generate_inputs(self):
        shape, dtype = self.shape, self.dtype
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        gamma = numpy.random.uniform(0.5, 1, shape[1]).astype(dtype)
        beta = numpy.random.uniform(-1, 1, shape[1]).astype(dtype)
        return x, gamma, beta

    def _get(self, device=None):
        mean, var = None, None
        if self.running_statistics:
            mean, var = self.mean.copy(), self.var.copy()
            if device is not None:
                mean, var = device.send((mean, var))
        return mean, var

    def forward(self, inputs, device):
        x, gamma, beta = inputs
        mean, var = self._get(device)
        return functions.instance_normalization(
            x, gamma, beta, running_mean=mean, running_var=var,
            eps=self.eps, decay=self.decay),

    def forward_expected(self, inputs):
        x, gamma, beta = inputs
        mean, var = self._get()
        y = _instance_normalization(
            x, gamma, beta, mean, var, self.eps, self.decay)
        return y,


testing.run_module(__name__, __file__)
