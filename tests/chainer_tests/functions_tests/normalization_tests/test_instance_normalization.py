import numpy
import unittest

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
    'contiguous': ['C', None],
    'running_statistics': [True, False],
})))
@backend.inject_backend_tests(
    None,
    # CPU tests
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
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    ])
class TestInstanceNormalization(testing.FunctionTestCase):

    def setUp(self):
        self.eps = 2e-5
        shape = self.shape
        dtype = self.dtype
        if self.running_statistics:
            mean = numpy.random.uniform(-1, 1, shape[1]).astype(dtype)
            var = numpy.random.uniform(-1, 1, shape[1]).astype(dtype)
        else:
            mean, var = None, None
        self.running_mean = mean
        self.running_var = var

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

    def forward(self, inputs, device):
        x, gamma, beta = inputs

        def _get_moving_avg(array, device, dtype):
            if array is None:
                return array
            else:
                new_array = array.copy()
                new_array = device.send_array(array).astype(dtype)
                return new_array

        running_mean = _get_moving_avg(self.running_mean, device, x.dtype)
        running_var = _get_moving_avg(self.running_var, device, x.dtype)
        return functions.instance_normalization(
            x, gamma, beta, running_mean=running_mean,
            running_var=running_var),

    def forward_expected(self, inputs):
        x, gamma, beta = inputs

        def _get_moving_avg(array):
            if array is None:
                return array
            else:
                return array.copy()

        running_mean = _get_moving_avg(self.running_mean)
        running_var = _get_moving_avg(self.running_var)
        y = _instance_normalization(
            x, gamma, beta, running_mean, running_var)
        return y,

    def before_test(self, test_name):
        if test_name == 'test_double_backward':
            if self.dtype == numpy.float16 and\
                    self.backend_config.use_cudnn == 'always':
                raise unittest.SkipTest


@testing.parameterize(*(testing.product({
    'shape': [(1, 4, 5, 5), (5, 4, 15)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'c_contiguous': ['C', None],
})))
@backend.inject_backend_tests(
    ['test_forward', 'test_backward', 'test_double_backward'],
    # CPU tests
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
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    ])
class TestFixedInstanceNormalization(testing.FunctionTestCase):

    def setUp(self):
        self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def generate_inputs(self):
        shape, dtype = self.shape, self.dtype
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        gamma = numpy.random.uniform(0, 1, shape[1]).astype(dtype)
        beta = numpy.random.uniform(-1, 1, shape[1]).astype(dtype)
        mean = numpy.random.uniform(-1, 1, shape[1]).astype(dtype)
        var = numpy.random.uniform(-1, 1, shape[1]).astype(dtype)
        return x, gamma, beta, mean, var

    def forward(self, inputs, device):
        x, gamma, beta, mean, var = inputs
        return functions.fixed_instance_normalization(
            x, gamma, beta, mean, var),

    def forward_expected(self, inputs):
        x, gamma, beta, mean, var = inputs
        y = _fixed_instance_normalization(x, gamma, beta, mean, var)
        return y,


testing.run_module(__name__, __file__)
