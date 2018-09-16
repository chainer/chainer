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


def _instance_normalization(args):
    for a in args:
        if isinstance(a, numpy.ndarray):
            print(a.shape)
            continue
        print(a)
    x, gamma, beta, mean, var, eps, expander = args
    N, C, *shape = x.shape
    org_shape = [N, C] + shape
    new_shape = [1, N * C] + shape

    xp = cuda.get_array_module(x)
    x = x.reshape(new_shape)
    gamma = xp.repeat(gamma, N, axis=0)
    beta = xp.repeat(beta, N, axis=0)
    mean = mean[expander]
    std = numpy.sqrt(var + eps)[expander]
    y_expect = (gamma[expander] * (x - mean) / std + beta[expander])
    return xp.reshape(y_expect, org_shape)


@testing.parameterize(*testing.product({
    'param_shape': [(3,)],
    'ndim': [0, 1, 2, 3],
    'eps': [2e-5],
    'use_running_statistics': [False, True],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'c_contiguous': [True, False],
}))
@backend.inject_backend_tests(
    ['test_forward', 'test_backward', 'test_double_backward'],
    # CPU tests
    testing.product({
        'use_cuda': [False],
        'use_ideep': 'never always'.split(' '),
    })
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': 'never always'.split(' '),
        'cudnn_fast_batch_normalization': [True, False],
    })
)
class TestInstanceNormalization(unittest.TestCase):

    def setUp(self):
        dtype = self.dtype
        param_shape = self.param_shape
        ndim = self.ndim

        x_shape = (5,) + param_shape + (2,) * ndim
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        gamma = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        beta = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        gy = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        ggx = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        gggamma = numpy.random.uniform(-1, 1, param_shape)
        ggbeta = numpy.random.uniform(-1, 1, param_shape)

        head_ndim = gamma.ndim + 1
        aggr_axes = (0,) + tuple(range(head_ndim, x.ndim))
        self.expander = (None, Ellipsis) + (None,) * ndim

        new_shape = (1,) + (5 * param_shape[0],) + (2,) * ndim
        _x = numpy.reshape(x, new_shape)
        mean = _x.mean(axis=aggr_axes)
        var = _x.var(axis=aggr_axes)

        self.decay, self.mean, self.var = 0.9, mean, var

        self.inputs = [x, gamma, beta]
        self.grad_outputs = [gy]
        self.grad_grad_inputs = [ggx, gggamma, ggbeta]

        self.in_options = {
            'decay': self.decay,
            'eps': self.eps,
            'use_running_statistics': self.use_running_statistics,
        }
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'dtype': numpy.float64}
        self.check_double_backward_options = {
            'dtype': numpy.float64, 'atol': 1e-3, 'rtol': 1e-2,
        }
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}

    def forward_cpu(self, inputs):
        y_expected = _instance_normalization(
            inputs + [self.mean, self.var, self.eps, self.expander]
        )
        return y_expected,

    def check_forward(self, inputs, backend_config):
        y_expected, = self.forward_cpu(inputs)

        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
        if not self.c_contiguous:
            inputs = _to_fcontiguous(inputs)

        with backend_config:
            y = functions.instance_normalization(
                *inputs, running_mean=None, running_var=None,
                **self.in_options
            )
        assert y.array.dtype == self.dtype

        testing.assert_allclose(
            y_expected, y.array, **self.check_forward_options
        )

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
            y = functions.instance_normalization(*inputs, **self.in_options)
            return y,

        with backend_config:
            gradient_check.check_backward(
                f, inputs, grad_outputs, **self.check_backward_options
            )

    def test_backward(self, backend_config):
        self.check_backward(self.inputs, self.grad_outputs, backend_config)

    def check_double_backward(
            self, inputs, grad_outputs, grad_grad_inputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)
            grad_grad_inputs = cuda.to_gpu(grad_grad_inputs)
        if not self.c_contiguous:
            inputs = _to_fcontiguous(inputs)
            grad_outputs = _to_fcontiguous(grad_outputs)
            grad_grad_inputs = _to_fcontiguous(grad_grad_inputs)

        def f(*inputs):
            return functions.instance_normalization(
                *inputs, **self.in_options)

        with backend_config:
            gradient_check.check_double_backward(
                f, inputs, grad_outputs, grad_grad_inputs,
                **self.check_double_backward_options)

    def test_double_backward(self, backend_config):
        self.check_double_backward(
            self.inputs, self.grad_outputs, self.grad_grad_inputs,
            backend_config)


@testing.parameterize(*(testing.product({
    'param_shape': [(3,)],
    'ndim': [0, 1, 2],
    'use_running_statistics': [False, True],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'c_contiguous': [True, False],
})))
@backend.inject_backend_tests(
    ['test_forward', 'test_backward', 'test_double_backward'],
    # CPU tests
    testing.product({
        'use_cuda': [False],
        'use_ideep': 'never always'.split(' '),
    })
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': 'never always'.split(' '),
        'cudnn_fast_batch_normalization': [True, False],
    })
)
class TestFixedInstanceNormalization(unittest.TestCase):

    def setUp(self):
        dtype = self.dtype
        param_shape = self.param_shape
        ndim = self.ndim

        x_shape = (5,) + param_shape + (2,) * ndim
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        gamma = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        beta = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        gy = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        ggx = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        gggamma = numpy.random.uniform(-1, 1, param_shape)
        ggbeta = numpy.random.uniform(-1, 1, param_shape)

        head_ndim = gamma.ndim + 1
        aggr_axes = (0,) + tuple(range(head_ndim, x.ndim))
        self.expander = (None, Ellipsis) + (None,) * ndim

        new_shape = (1,) + (5 * param_shape[0],) + (2,) * ndim
        _x = numpy.reshape(x, new_shape)
        mean = _x.mean(axis=aggr_axes)
        var = _x.var(axis=aggr_axes)
        ggmean = numpy.random.uniform(-1, 1, mean.shape).astype(dtype)
        ggvar = numpy.random.uniform(-1, 1, var.shape).astype(dtype)

        self.decay, self.mean, self.var = 0.9, mean, var

        self.inputs = [x, gamma, beta, mean, var]
        self.grad_outputs = [gy]
        self.grad_grad_inputs = [ggx, gggamma, ggbeta, ggmean, ggvar]

        self.in_options = {
            'decay': self.decay,
            'eps': self.eps,
            'use_running_statistics': self.use_running_statistics,
        }
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'dtype': numpy.float64}
        self.check_double_backward_options = {
            'dtype': numpy.float64, 'atol': 1e-3, 'rtol': 1e-2,
        }
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}

    def forward_cpu(self, inputs):
        y_expected = _instance_normalization(
            inputs + [self.eps, self.expander]
        )
        return y_expected,

    def check_forward(self, inputs, backend_config):
        y_expected, = self.forward_cpu(inputs)

        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
        if not self.c_contiguous:
            inputs = _to_fcontiguous(inputs)

        with backend_config:
            y = functions.fixed_instance_normalization(*inputs, eps=self.eps)
        assert y.data.dtype == self.dtype

        testing.assert_allclose(
            y_expected, y.data, **self.check_forward_options
        )

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
            y = functions.fixed_instance_normalization(*inputs, eps=self.eps)
            return y,

        with backend_config:
            gradient_check.check_backward(
                f, inputs, grad_outputs, **self.check_backward_options
            )

    def test_backward(self, backend_config):
        self.check_backward(self.inputs, self.grad_outputs, backend_config)

    def check_double_backward(
            self, inputs, grad_outputs, grad_grad_inputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)
            grad_grad_inputs = cuda.to_gpu(grad_grad_inputs)
        if not self.c_contiguous:
            inputs = _to_fcontiguous(inputs)
            grad_outputs = _to_fcontiguous(grad_outputs)
            grad_grad_inputs = _to_fcontiguous(grad_grad_inputs)

        def f(*inputs):
            return functions.fixed_instance_normalization(
                *inputs, eps=self.eps)

        with backend_config:
            gradient_check.check_double_backward(
                f, inputs, grad_outputs, grad_grad_inputs,
                **self.check_double_backward_options)

    def test_double_backward(self, backend_config):
        self.check_double_backward(
            self.inputs, self.grad_outputs, self.grad_grad_inputs,
            backend_config)


testing.run_module(__name__, __file__)
