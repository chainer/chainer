import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import backend


def _calc_instancenorm_stats(x, aggr_axes):
    b, c = x.shape[:2]
    new_shape = (1, b * c) + x.shape[2:]
    x = x.reshape(new_shape)
    mean = x.mean(aggr_axes)
    var = x.var(aggr_axes)
    return mean, var


def _as_noncontiguous_array(array):
    # TODO(niboshi): cupy + cudnn test fails in F.fixed_batch_normalization.
    # Fix it and use testing.array._as_noncontiguous_array.
    def as_noncontiguous_array(arr):
        if isinstance(arr, (numpy.ndarray, cuda.ndarray)):
            if arr is None:
                return None
            xp = chainer.backend.get_array_module(arr)
            return xp.asfortranarray(arr)
        return testing.array._as_noncontiguous_array(arr)

    if isinstance(array, (list, tuple)):
        return type(array)([as_noncontiguous_array(arr) for arr in array])
    return as_noncontiguous_array(array)


def _instance_normalization(
        inputs, running_mean=None, running_var=None, decay=None):
    # shape| x: NCHW, gamma N*C, beta N*C, mean C, var C
    x, gamma, beta, mean, var, eps, expander = inputs
    xp = chainer.backend.get_array_module(x)
    original_shape = x.shape
    b, c = x.shape[:2]
    x_ = x.reshape((1, b * c) + original_shape[2:])
    mean_expanded = mean[expander]
    std = numpy.sqrt(var + eps)[expander]
    gamma = xp.concatenate([gamma] * b)
    beta = xp.concatenate([beta] * b)
    assert b * c == gamma.size and b * c == beta.size
    y_expect = (gamma[expander] * (x_ - mean_expanded) / std + beta[expander])

    if running_mean is not None or running_var is not None:
        m = x.size // gamma.size
        adjust = m / max(m - 1., 1.)  # unbiased estimation
        if running_mean is not None:
            running_mean_ = xp.concatenate([running_mean] * b)
            running_mean_ *= decay
            running_mean_ += (1 - decay) * mean
            running_mean[:] = running_mean_.reshape((b, c)).mean(axis=0)
        if running_var is not None:
            running_var_ = xp.concatenate([running_var] * b)
            running_var_ *= decay
            running_var_ += (1 - decay) * adjust * var
            running_var[:] = running_var_.reshape((b, c)).mean(axis=0)
    return y_expect.reshape(original_shape)


@testing.parameterize(*(testing.product({
    'shape': [(1, 4, 5, 5), (5, 4, 15)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'c_contiguous': [True, False],
    'running_statistics': [True, False],
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
class TestInstanceNormalization(unittest.TestCase):

    def setUp(self):
        dtype = self.dtype
        param_shape = 4
        shape = self.shape

        gamma = numpy.random.uniform(.5, 1, param_shape).astype(dtype)
        beta = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
        ggx = numpy.random.uniform(-1, 1, shape).astype(dtype)
        gggamma = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        ggbeta = numpy.random.uniform(-1, 1, param_shape).astype(dtype)

        if self.running_statistics:
            self.running_mean = numpy.random.uniform(
                -1, 1, param_shape).astype(dtype)
            self.running_var = numpy.random.uniform(
                -1, 1, param_shape).astype(dtype)
        else:
            self.running_mean = None
            self.running_var = None

        head_ndim = gamma.ndim + 1
        aggr_axes = (0,) + tuple(range(head_ndim, x.ndim))
        self.expander = tuple(
            None if i in aggr_axes else slice(None)
            for i in range(x.ndim)
        )

        self.mean, self.var = _calc_instancenorm_stats(x, aggr_axes)
        self.decay = 0.9
        self.eps = 2e-5

        self.inputs = [x, gamma, beta]
        self.grad_outputs = [gy]
        self.grad_grad_inputs = [ggx, gggamma, ggbeta]

        if self.running_statistics:
            self.running_mean = numpy.random.uniform(
                -1, 1, param_shape).astype(dtype)
            self.running_var = numpy.random.uniform(
                -1, 1, param_shape).astype(dtype)
        else:
            self.running_mean = None
            self.running_var = None

        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}

    def forward_cpu(self, inputs, running_mean=None, running_var=None):
        y_expect = _instance_normalization(
            inputs + [self.mean, self.var, self.eps, self.expander],
            running_mean, running_var, self.decay
        )
        return y_expect,

    def check_forward(self, inputs, backend_config):
        if backend_config.use_chainerx and self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')
        if self.running_statistics:
            running_mean_expected = self.running_mean.copy()
            running_var_expected = self.running_var.copy()
        else:
            running_mean_expected = None
            running_var_expected = None

        y_expected, = self.forward_cpu(
            inputs, running_mean_expected, running_var_expected)
        inputs = backend_config.get_array(self.inputs)
        running_mean = backend_config.get_array(self.running_mean)
        running_var = backend_config.get_array(self.running_var)

        if not self.c_contiguous:
            inputs = _as_noncontiguous_array(inputs)
            running_mean = _as_noncontiguous_array(running_mean)
            running_var = _as_noncontiguous_array(running_var)
        with backend_config:
            y = functions.instance_normalization(
                inputs[0], inputs[1], inputs[2],
                running_mean=running_mean,
                running_var=running_var,
                decay=self.decay
            )

        assert y.array.dtype == self.dtype

        testing.assert_allclose(
            y_expected, y.array, **self.check_forward_options)
        if self.running_statistics:
            testing.assert_allclose(
                running_mean_expected, running_mean,
                **self.check_forward_options)
            testing.assert_allclose(
                running_var_expected, running_var,
                **self.check_forward_options)

    def test_forward(self, backend_config):
        self.check_forward(self.inputs, backend_config)

    def check_backward(self, inputs, grad_outputs, backend_config):
        if backend_config.use_chainerx and self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')
        inputs = backend_config.get_array(inputs)
        grad_outputs = backend_config.get_array(grad_outputs)
        if not self.c_contiguous:
            inputs = _as_noncontiguous_array(inputs)
            grad_outputs = _as_noncontiguous_array(grad_outputs)

        def f(*inputs):
            y = functions.instance_normalization(*inputs, eps=self.eps)
            return y,

        with backend_config:
            gradient_check.check_backward(
                f, inputs, grad_outputs,
                **self.check_backward_options
            )

    def test_backward(self, backend_config):
        self.check_backward(self.inputs, self.grad_outputs, backend_config)

    def check_double_backward(self, inputs, grad_outputs, grad_grad_inputs,
                              backend_config):
        if backend_config.use_chainerx and self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')
        inputs = backend_config.get_array(inputs)
        grad_outputs = backend_config.get_array(grad_outputs)
        grad_grad_inputs = backend_config.get_array(grad_grad_inputs)
        if not self.c_contiguous:
            inputs = _as_noncontiguous_array(inputs)
            grad_outputs = _as_noncontiguous_array(grad_outputs)
            grad_grad_inputs = _as_noncontiguous_array(grad_grad_inputs)

        def f(*inputs):
            return functions.instance_normalization(*inputs, eps=self.eps)

        with backend_config:
            gradient_check.check_double_backward(
                f, inputs, grad_outputs, grad_grad_inputs,
                **self.check_double_backward_options)

    def test_double_backward(self, backend_config):
        self.check_double_backward(self.inputs, self.grad_outputs,
                                   self.grad_grad_inputs, backend_config)


@testing.parameterize(*(testing.product({
    'shape': [(2, 4, 5, 5), (5, 4, 15)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'c_contiguous': [True, False],
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
class TestFixedInstanceNormalization(unittest.TestCase):

    def setUp(self):
        shape = self.shape
        param_shape = 4
        dtype = self.dtype

        gamma = numpy.random.uniform(.5, 1, param_shape).astype(dtype)
        beta = numpy.random.uniform(.5, 1, param_shape).astype(dtype)
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        mean = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        tiled_mean = numpy.concatenate([mean] * x.shape[0])
        var = numpy.random.uniform(.5, 1, param_shape).astype(dtype)
        tiled_var = numpy.concatenate([var] * x.shape[0])

        gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
        ggx = numpy.random.uniform(-1, 1, shape).astype(dtype)
        gggamma = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        ggbeta = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        ggmean = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        ggvar = numpy.random.uniform(-1, 1, param_shape).astype(dtype)

        self.decay = 0.0
        self.eps = 2e-5
        self.expander = (None, Ellipsis) + (None,) * (len(shape) - 2)
        self.inputs = [x, gamma, beta, tiled_mean, tiled_var]
        self.f_inputs = [x, gamma, beta, mean, var]
        self.grad_outputs = [gy]
        self.grad_grad_inputs = [ggx, gggamma, ggbeta, ggmean, ggvar]

        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'dtype': numpy.float64}
        self.check_double_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}

    def forward_cpu(self, inputs):
        y_expect = _instance_normalization(inputs + [self.eps, self.expander])
        return y_expect,

    def check_forward(self, inputs, f_inputs, enable_backprop, backend_config):
        if backend_config.use_chainerx and self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')

        y_expected, = self.forward_cpu(inputs)

        f_inputs = backend_config.get_array(f_inputs)
        if not self.c_contiguous:
            f_inputs = _as_noncontiguous_array(f_inputs)

        with chainer.using_config('enable_backprop', enable_backprop):
            with backend_config:
                y = functions.fixed_instance_normalization(
                    *f_inputs, eps=self.eps)

        assert y.array.dtype == self.dtype

        testing.assert_allclose(
            y_expected, y.array, **self.check_forward_options)

    def test_forward(self, backend_config):
        self.check_forward(self.inputs, self.f_inputs, True, backend_config)

    # TODO(crcrpar): Fix the below error:
    # > TypeError: test_forward_with_enable_backprop() missing 1 required
    # > positional argument: 'backend_config'
    def test_forward_with_enable_backprop(self, backend_config):
        self.check_forward(self.inputs, self.f_inputs, False, backend_config)

    def check_backward(self, inputs, grad_outputs, backend_config):
        # TODO(niboshi): Support it
        if backend_config.use_chainerx and self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')

        inputs = backend_config.get_array(inputs)
        grad_outputs = backend_config.get_array(grad_outputs)
        if not self.c_contiguous:
            inputs = _as_noncontiguous_array(inputs)
            grad_outputs = _as_noncontiguous_array(grad_outputs)

        def f(*inputs):
            y = functions.fixed_instance_normalization(*inputs, eps=self.eps)
            return y,

        with backend_config:
            gradient_check.check_backward(
                f, inputs, grad_outputs,
                **self.check_backward_options)

    def test_backward(self, backend_config):
        self.check_backward(self.f_inputs, self.grad_outputs, backend_config)

    def check_double_backward(
            self, inputs, grad_outputs, grad_grad_inputs, backend_config):
        # TODO(niboshi): Support it
        if backend_config.use_chainerx and self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')

        inputs = backend_config.get_array(inputs)
        grad_outputs = backend_config.get_array(grad_outputs)
        grad_grad_inputs = backend_config.get_array(grad_grad_inputs)
        if not self.c_contiguous:
            inputs = _as_noncontiguous_array(inputs)
            grad_outputs = _as_noncontiguous_array(grad_outputs)
            grad_grad_inputs = _as_noncontiguous_array(grad_grad_inputs)

        def f(*inputs):
            return functions.fixed_instance_normalization(
                *inputs, eps=self.eps)

        with backend_config:
            gradient_check.check_double_backward(
                f, inputs, grad_outputs, grad_grad_inputs,
                **self.check_double_backward_options)

    def test_double_backward(self, backend_config):
        self.check_double_backward(
            self.f_inputs, self.grad_outputs, self.grad_grad_inputs,
            backend_config)


testing.run_module(__name__, __file__)
