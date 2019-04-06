import unittest

import numpy
import pytest
import six
import warnings

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend


def _batch_normalization(
        inputs, running_mean=None, running_var=None, decay=None):
    x, gamma, beta, mean, var, eps, expander = inputs
    mean_expanded = mean[expander]
    std = numpy.sqrt(var + eps, dtype=x.dtype)[expander]
    y_expect = gamma[expander] * (x - mean_expanded) / std + beta[expander]

    if running_mean is not None or running_var is not None:
        m = x.size // gamma.size
        adjust = m / max(m - 1., 1.)  # unbiased estimation
        if running_mean is not None:
            running_mean *= decay
            running_mean += (1 - decay) * mean
        if running_var is not None:
            running_var *= decay
            running_var += (1 - decay) * adjust * var

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
        'running_statistics': [True, False],
    }),
) + testing.product({
    'param_shape': [(3,)],
    'ndim': [1],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'running_statistics': [True, False],
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
    }))
class TestBatchNormalization(testing.FunctionTestCase):

    def setUp(self):
        self.check_forward_options.update({'atol': 1e-4, 'rtol': 1e-3})
        self.check_double_backward_options.update({'atol': 1e-3, 'rtol': 1e-2})
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 1e-2, 'rtol': 1e-2})
            self.check_backward_options.update({'atol': 1e-2, 'rtol': 1e-2})
            self.check_double_backward_options.update(
                {'atol': 1e-2, 'rtol': 1e-2})

        self._prepare_shape_and_expander()

        if self.running_statistics:
            self.running_mean = numpy.random.uniform(
                -1, 1, self.param_shape).astype(self.dtype)
            self.running_var = numpy.random.uniform(
                -1, 1, self.param_shape).astype(self.dtype)
        else:
            self.running_mean = None
            self.running_var = None

    def _prepare_shape_and_expander(self):
        self.decay = 0.9
        self.bn_options = {'decay': self.decay, 'eps': self.eps}
        if not hasattr(self, 'axis'):
            param_shape = self.param_shape
            self.input_shape = (5,) + param_shape + (2,) * self.ndim
            head_ndim = len(param_shape) + 1
            self.aggr_axes = (0,) + tuple(
                six.moves.range(head_ndim, len(self.input_shape)))
            self.expander = (None, Ellipsis) + (None,) * self.ndim
        else:
            self.bn_options['axis'] = self.axis
            aggr_axes = self.axis
            if isinstance(self.axis, int):
                aggr_axes = (aggr_axes,)
            self.aggr_axes = aggr_axes
            self.param_shape = tuple(
                s for i, s in enumerate(self.input_shape)
                if i not in aggr_axes)
            self.expander = tuple(
                None if i in aggr_axes else slice(None)
                for i in range(len(self.input_shape)))

    def generate_inputs(self):
        input_shape = self.input_shape
        param_shape = self.param_shape
        gamma = numpy.random.uniform(.5, 1, param_shape).astype(self.dtype)
        beta = numpy.random.uniform(-1, 1, param_shape).astype(self.dtype)
        x = numpy.random.uniform(-1, 1, input_shape).astype(self.dtype)
        self.mean = x.mean(axis=self.aggr_axes)
        self.var = x.var(axis=self.aggr_axes)
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
        y = functions.batch_normalization(
            x, gamma, beta, running_mean=mean, running_var=var,
            **self.bn_options)
        if self.running_statistics:
            return y, chainer.Variable(mean), chainer.Variable(var)
        return y,

    def forward_expected(self, inputs):
        running_mean, running_var = self.get_running_averages()
        y_expect = _batch_normalization(
            inputs + (self.mean, self.var, self.eps, self.expander),
            running_mean, running_var, self.decay)
        if self.running_statistics:
            return y_expect, running_mean, running_var,
        return y_expect,

    def before_test(self, test_name):
        if 'backward' in test_name and self.running_statistics:
            raise unittest.SkipTest()


@testing.parameterize(*(testing.product({
    'param_shape': [(3,), (3, 4), (3, 2, 3)],
    'ndim': [0, 1, 2],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float32],
}) + testing.product({
    'param_shape': [(3,)],
    'ndim': [1],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
@backend.inject_backend_tests(
    None,
    # CPU tests
    [{}]
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
    }))
class TestFixedBatchNormalization(testing.FunctionTestCase):

    def setUp(self):
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options.update({'atol': 1e-2, 'rtol': 1e-2})
            self.check_double_backward_options.update({
                'atol': 1e-2, 'rtol': 1e-2})

        self.decay = 0.0
        self.expander = (None, Ellipsis) + (None,) * self.ndim

    def generate_inputs(self):
        param_shape = self.param_shape
        dtype = self.dtype
        ndim = self.ndim

        gamma = numpy.random.uniform(.5, 1, param_shape).astype(dtype)
        beta = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        shape = (5,) + param_shape + (2,) * ndim
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        mean = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        var = numpy.random.uniform(0.5, 1, param_shape).astype(dtype)

        return x, gamma, beta, mean, var

    def forward(self, inputs, device):
        y = functions.fixed_batch_normalization(*inputs, eps=self.eps)
        return y,

    def forward_expected(self, inputs):
        y_expect = _batch_normalization(inputs + (self.eps, self.expander))
        return y_expect,


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
        self.gamma = cuda.cupy.random.uniform(
            .5, 1, param_shape).astype(self.dtype)
        self.beta = cuda.cupy.random.uniform(
            -1, 1, param_shape).astype(self.dtype)
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
                    'cupy.cudnn.batch_normalization_forward_training'
            ) as func:
                self.forward()
                assert func.called == self.expect

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            y.grad = self.gy
            with testing.patch(
                    'cupy.cudnn.batch_normalization_backward'
            ) as func:
                y.backward()
                assert func.called == self.expect


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
        eps = -0.1
        if chainer.backends.cuda.libcudnn.get_build_version() < 7500:
            eps = 2e-6
        with pytest.raises(RuntimeError):
            functions.batch_normalization(*self.args, eps=eps)


@attr.cudnn
class TestFixedBatchNormalizationCudnnEps(unittest.TestCase):
    def setUp(self):
        ndim = 0
        param_shape = (3,)
        dtype = numpy.float32
        gamma = cuda.cupy.random.uniform(.5, 1, param_shape).astype(dtype)
        beta = cuda.cupy.random.uniform(-1, 1, param_shape).astype(dtype)
        mean = cuda.cupy.random.uniform(-1, 1, param_shape).astype(dtype)
        var = cuda.cupy.random.uniform(-1, 1, param_shape).astype(dtype)
        shape = (7,) + param_shape + (2,) * ndim
        x = cuda.cupy.random.uniform(-1, 1, shape).astype(dtype)
        self.args = [x, gamma, beta, mean, var]

    def test_valid(self):
        functions.fixed_batch_normalization(*self.args, eps=1e-5)

    def test_invalid(self):
        eps = -0.1
        if chainer.backends.cuda.libcudnn.get_build_version() < 7500:
            eps = 2e-6
        with pytest.raises(RuntimeError):
            functions.fixed_batch_normalization(*self.args, eps=eps)


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
