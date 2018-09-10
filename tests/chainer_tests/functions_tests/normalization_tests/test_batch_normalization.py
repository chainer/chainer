import unittest

import numpy
import six
import warnings

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend


def _to_fcontiguous(arrays):
    xp = cuda.get_array_module(*arrays)
    return [xp.asfortranarray(a) for a in arrays]


def _batch_normalization(args):
    x, gamma, beta, mean, var, eps, expander = args
    mean = mean[expander]
    std = numpy.sqrt(var + eps)[expander]
    y_expect = (gamma[expander] * (x - mean) / std + beta[expander])
    return y_expect


@testing.parameterize(*(testing.product_dict(
    testing.product({
        'param_shape': [(3,), (3, 4), (3, 2, 3)],
        'ndim': [0, 1, 2],
    }) + [
        {'input_shape': (5, 4, 3, 2), 'axis': (0, 2, 3)},
        {'input_shape': (5, 4), 'axis': 0},
        {'input_shape': (5, 4, 3), 'axis': (0, 1)},
    ],
    testing.product({
        'dtype': [numpy.float32],
        'eps': [2e-5, 5e-1],
        'c_contiguous': [True, False],
    }),
) + testing.product({
    'param_shape': [(3,)],
    'ndim': [1],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'c_contiguous': [True, False],
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
    }))
class TestBatchNormalization(unittest.TestCase):

    def setUp(self):
        dtype = self.dtype

        if not hasattr(self, 'axis'):
            param_shape = self.param_shape
            ndim = self.ndim
            shape = (5,) + param_shape + (2,) * ndim
        else:
            aggr_axes = self.axis
            if isinstance(self.axis, int):
                aggr_axes = self.axis,
            param_shape = tuple(
                s
                for i, s in enumerate(self.input_shape)
                if i not in aggr_axes
            )
            shape = self.input_shape

        gamma = numpy.random.uniform(.5, 1, param_shape).astype(dtype)
        beta = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
        ggx = numpy.random.uniform(-1, 1, shape).astype(dtype)
        gggamma = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        ggbeta = numpy.random.uniform(-1, 1, param_shape).astype(dtype)

        if not hasattr(self, 'axis'):
            head_ndim = gamma.ndim + 1
            aggr_axes = (0,) + tuple(six.moves.range(head_ndim, x.ndim))

            self.expander = (None, Ellipsis) + (None,) * ndim
        else:
            self.expander = tuple(
                None if i in aggr_axes else slice(None)
                for i in range(x.ndim)
            )

        mean = x.mean(axis=aggr_axes)
        var = x.var(axis=aggr_axes)

        self.decay = 0.9
        self.mean = mean
        self.var = var

        self.inputs = [x, gamma, beta]
        self.grad_outputs = [gy]
        self.grad_grad_inputs = [ggx, gggamma, ggbeta]

        self.bn_options = {
            'decay': self.decay,
            'eps': self.eps,
        }
        if hasattr(self, 'axis'):
            self.bn_options['axis'] = self.axis
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'dtype': numpy.float64}
        self.check_double_backward_options = {
            'dtype': numpy.float64, 'atol': 1e-3, 'rtol': 1e-2}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}

    def forward_cpu(self, inputs):
        y_expect = _batch_normalization(
            inputs + [self.mean, self.var, self.eps, self.expander])
        return y_expect,

    def check_forward(self, inputs, backend_config):
        y_expected, = self.forward_cpu(inputs)

        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
        if not self.c_contiguous:
            inputs = _to_fcontiguous(inputs)

        with backend_config:
            y = functions.batch_normalization(
                *inputs, running_mean=None,
                running_var=None, **self.bn_options)
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
            y = functions.batch_normalization(
                *inputs, **self.bn_options)
            return y,

        with backend_config:
            gradient_check.check_backward(
                f, inputs, grad_outputs,
                **self.check_backward_options)

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
            return functions.batch_normalization(
                *inputs, **self.bn_options)

        with backend_config:
            gradient_check.check_double_backward(
                f, inputs, grad_outputs, grad_grad_inputs,
                **self.check_double_backward_options)

    def test_double_backward(self, backend_config):
        self.check_double_backward(
            self.inputs, self.grad_outputs, self.grad_grad_inputs,
            backend_config)


@testing.parameterize(*(testing.product({
    'param_shape': [(3,), (3, 4), (3, 2, 3)],
    'ndim': [0, 1, 2],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float32],
    'c_contiguous': [True, False],
}) + testing.product({
    'param_shape': [(3,)],
    'ndim': [1],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'c_contiguous': [True, False],
})))
@backend.inject_backend_tests(
    ['test_forward', 'test_backward', 'test_double_backward'],
    # CPU tests
    [{'use_cuda': False}]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
        'cudnn_fast_batch_normalization': [True, False],
    }))
class TestFixedBatchNormalization(unittest.TestCase):

    def setUp(self):
        param_shape = self.param_shape
        dtype = self.dtype
        ndim = self.ndim

        gamma = numpy.random.uniform(.5, 1, param_shape).astype(dtype)
        beta = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        shape = (5,) + param_shape + (2,) * ndim
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        mean = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        var = numpy.random.uniform(0.5, 1, param_shape).astype(dtype)

        gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
        ggx = numpy.random.uniform(-1, 1, shape).astype(dtype)
        gggamma = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        ggbeta = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        ggmean = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        ggvar = numpy.random.uniform(-1, 1, param_shape).astype(dtype)

        self.decay = 0.0
        self.expander = (None, Ellipsis) + (None,) * ndim

        self.inputs = [x, gamma, beta, mean, var]
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
        y_expect = _batch_normalization(inputs + [self.eps, self.expander])
        return y_expect,

    def check_forward(self, inputs, backend_config):
        y_expected, = self.forward_cpu(inputs)

        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
        if not self.c_contiguous:
            inputs = _to_fcontiguous(inputs)

        with backend_config:
            y = functions.fixed_batch_normalization(*inputs, eps=self.eps)
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
            y = functions.fixed_batch_normalization(*inputs, eps=self.eps)
            return y,

        with backend_config:
            gradient_check.check_backward(
                f, inputs, grad_outputs,
                **self.check_backward_options)

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
            return functions.fixed_batch_normalization(*inputs, eps=self.eps)

        with backend_config:
            gradient_check.check_double_backward(
                f, inputs, grad_outputs, grad_grad_inputs,
                **self.check_double_backward_options)

    def test_double_backward(self, backend_config):
        self.check_double_backward(
            self.inputs, self.grad_outputs, self.grad_grad_inputs,
            backend_config)


@testing.parameterize(*testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
    'eps': [2e-5, 5e-1],
    # TODO(bkvogel): Check float16 support again in next cuDNN version.
    'dtype': [numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestBatchNormalizationCudnnCall(unittest.TestCase):

    def setUp(self):
        ndim = 0
        param_shape = (3,)
        self.gamma = cuda.cupy.random.uniform(.5, 1,
                                              param_shape).astype(self.dtype)
        self.beta = cuda.cupy.random.uniform(-1, 1,
                                             param_shape).astype(self.dtype)
        shape = (7,) + param_shape + (2,) * ndim
        self.x = cuda.cupy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.args = [self.x, self.gamma, self.beta]
        head_ndim = self.gamma.ndim + 1
        self.aggr_axes = (0,) + tuple(six.moves.range(head_ndim, self.x.ndim))
        self.mean = self.x.mean(axis=self.aggr_axes)
        self.var = self.x.var(axis=self.aggr_axes) + self.eps
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.expect = chainer.should_use_cudnn('>=auto', 5000)

    def forward(self):
        return functions.batch_normalization(
            *[chainer.Variable(i) for i in self.args], eps=self.eps,
            running_mean=self.mean, running_var=self.var)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with testing.patch(
                    'cupy.cuda.cudnn.batchNormalizationForwardTraining'
            ) as func:
                self.forward()
                self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            y.grad = self.gy
            with testing.patch(
                    'cupy.cuda.cudnn.batchNormalizationBackward'
            ) as func:
                y.backward()
                self.assertEqual(func.called, self.expect)


@attr.cudnn
class TestBatchNormalizationCudnnEps(unittest.TestCase):
    def setUp(self):
        ndim = 0
        param_shape = (3,)
        dtype = numpy.float32
        gamma = cuda.cupy.random.uniform(.5, 1, param_shape).astype(dtype)
        beta = cuda.cupy.random.uniform(-1, 1, param_shape).astype(dtype)
        shape = (7,) + param_shape + (2,) * ndim
        x = cuda.cupy.random.uniform(-1, 1, shape).astype(dtype)
        self.args = [x, gamma, beta]

    def test_valid(self):
        functions.batch_normalization(*self.args, eps=1e-5)

    def test_invalid(self):
        with self.assertRaises(RuntimeError):
            functions.batch_normalization(*self.args, eps=2e-6)


class TestBatchNormalizationWarning(unittest.TestCase):
    def setUp(self):
        pass

    def create_batch(self, param_shape, x_shape):
        dtype = numpy.float32
        gamma = numpy.random.uniform(.5, 1, param_shape).astype(dtype)
        beta = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        args = [x, gamma, beta]
        return args

    def test_invalid_batch(self):
        args = self.create_batch((3,), (1, 3))
        with testing.assert_warns(UserWarning):
            functions.batch_normalization(*args)

    def test_invalid_batch_no_batch_axis(self):
        args = self.create_batch((1, 3,), (1, 3, 1))
        with testing.assert_warns(UserWarning):
            functions.batch_normalization(*args, axis=2)

    def test_valid_batch(self):
        args = self.create_batch((3,), (1, 3, 2, 2))
        with warnings.catch_warnings(record=True) as w:
            functions.batch_normalization(*args)
            assert len(w) == 0

    def test_valid_batch_no_batch_axis(self):
        args = self.create_batch((1, 3,), (1, 3, 2))
        with warnings.catch_warnings(record=True) as w:
            functions.batch_normalization(*args, axis=2)
            assert len(w) == 0


testing.run_module(__name__, __file__)
