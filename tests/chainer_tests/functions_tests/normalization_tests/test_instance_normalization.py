import numpy
import unittest

import chainer
from chainer import functions
from chainer import testing
from chainer.testing import backend


def _instance_normalization(
        x, gamma, beta, running_mean=None, running_var=None,
        eps=2e-5, decay=0.9):
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
    std = numpy.sqrt(var + eps, dtype=x.dtype)
    x_reshaped_normalized = (x_reshaped - mean[expander]) / std[expander]
    x_normalized = x_reshaped_normalized.reshape(org_shape)
    y = gamma[expander] * x_normalized + beta[expander]
    if running_mean is not None or running_var is not None:
        m = x.size // c
        adjust = m / max(m - 1., 1.)  # unbiased estimation
        if running_mean is not None:
            running_mean *= decay
            running_mean += (1 - decay) * mean.reshape(b, c).mean(axis=0)
        if running_var is not None:
            running_var *= decay
            running_var += (
                1 - decay) * adjust * var.reshape(b, c).mean(axis=0)
    return y


def _fixed_instance_normalization(
        x, gamma, beta, mean, var, eps=2e-5):
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
    # CPU tests
    testing.product({
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
        self.eps = 2e-5
        shape = self.shape
        dtype = self.dtype
        if self.running_statistics:
            self.running_mean = numpy.random.uniform(
                -1, 1, shape[1]).astype(dtype)
            self.running_var = numpy.random.uniform(
                -1, 1, shape[1]).astype(dtype)

        self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        if dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def generate_inputs(self):
        shape, dtype = self.shape, self.dtype
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        gamma = numpy.random.uniform(0.5, 1, shape[1]).astype(dtype)
        beta = numpy.random.uniform(-1, 1, shape[1]).astype(dtype)
        return x, gamma, beta

    def get_running_averages(self, device=None):
        if not self.running_statistics:
            return None, None
        mean = self.running_mean.copy()
        var = self.running_var.copy()
        if device is None:
            return mean, var
        return device.send((mean, var))

    def forward(self, inputs, device):
        x, gamma, beta = inputs
        mean, var = self.get_running_averages(device)
        y = functions.instance_normalization(
            x, gamma, beta, running_mean=mean, running_var=var,
            eps=self.eps, decay=self.decay)
        if not self.running_statistics:
            return y,
        return y, chainer.Variable(mean), chainer.Variable(var)

    def forward_expected(self, inputs):
        x, gamma, beta = inputs
        running_mean, running_var = self.get_running_averages()
        y = _instance_normalization(
            x, gamma, beta, running_mean, running_var, self.eps, self.decay)
        if not self.running_statistics:
            return y,
        return y, running_mean, running_var

    def before_test(self, test_name):
        if 'backward' in test_name and self.running_statistics:
            raise unittest.SkipTest()


@testing.parameterize(*(testing.product({
    'shape': [(1, 4, 5, 5), (5, 4, 15)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'eps': [2e-5],
})))
@backend.inject_backend_tests(
    None,
    # # CPU tests
    testing.product({
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
class TestFixedInstanceNormalization(testing.FunctionTestCase):

    def setUp(self):
        self.check_forward_options.update({'atol': 1e-4, 'rtol': 1e-3})
        self.check_backward_options.update({'atol': 1e-5, 'rtol': 1e-4})
        self.check_double_backward_options.update({'atol': 1e-3, 'rtol': 1e-2})
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 1e-2, 'rtol': 1e-2})
            self.check_backward_options.update({'atol': 1e-2, 'rtol': 1e-2})
            self.check_double_backward_options.update(
                {'atol': 1e-2, 'rtol': 1e-2})

    def generate_inputs(self):
        shape, dtype = self.shape, self.dtype
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        gamma = numpy.random.uniform(-1, 1, shape[1]).astype(dtype)
        beta = numpy.random.uniform(-1, 1, shape[1]).astype(dtype)
        mean = numpy.random.uniform(-1, 1, shape[1]).astype(dtype)
        var = numpy.random.uniform(.5, 1, shape[1]).astype(dtype)
        return x, gamma, beta, mean, var

    def forward(self, inputs, device):
        x, gamma, beta, mean, var = inputs
        return functions.fixed_instance_normalization(
            x, gamma, beta, mean, var, eps=self.eps),

    def forward_expected(self, inputs):
        x, gamma, beta, mean, var = inputs
        y = _fixed_instance_normalization(x, gamma, beta, mean, var, self.eps)
        return y,


testing.run_module(__name__, __file__)
