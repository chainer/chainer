import unittest

import functools
import numpy
from operator import mul

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import testing
from chainer.testing import attr
from chainer.utils import conv


@testing.parameterize(*(testing.product({
    'dims': [(5,), (4, 3), (3, 4, 3)],
    'dilate': [1, 2],
    'groups': [1, 2],
    'cover_all': [True, False],
    'contiguous': ['C'],
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
    'b_dtype': [numpy.float32],
    'autotune': [True, False],
    'nobias': [True, False],
}) + testing.product({
    'dims': [(4,)],
    'dilate': [1],
    'groups': [1],
    'cover_all': [False],
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'b_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'autotune': [False],
    'nobias': [True, False],
})))
@testing.inject_backend_tests(
    ['test_forward', 'test_backward', 'test_double_backward',
     'test_consistency_forward', 'test_consistency_regression_forward'],
    # CPU tests
    [{}]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
    })
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    ]
)
class TestConvolutionND(testing.FunctionTestCase):

    def setUp(self):
        self.N = 2
        self.in_channels = 4
        self.out_channels = 2
        self.ndim = len(self.dims)
        self.ksize = (2,) * self.ndim
        self.stride = (1,) * self.ndim
        self.pad = (1,) * self.ndim
        self.dilate = (self.dilate,) * self.ndim

        self.x_shape = (self.N, self.in_channels) + self.dims
        self.W_shape = (
            self.out_channels, self.in_channels // self.groups) + self.ksize
        self.W_scale = numpy.sqrt(
            1. / functools.reduce(mul, self.ksize, self.in_channels))
        self.gy_shape = (self.N, self.out_channels) + tuple(
            conv.get_conv_outsize(d, k, s, p, cover_all=self.cover_all, d=di)
            for (d, k, s, p, di)
            in zip(self.dims, self.ksize, self.stride, self.pad, self.dilate))

        self.check_backward_options.update({'atol': 5e-5, 'rtol': 5e-4})
        self.check_double_backward_options.update(
            {'atol': 5e-4, 'rtol': 5e-3})
        if self.x_dtype == numpy.float16 or self.W_dtype == numpy.float16:
            self.check_forward_options.update({'atol': 5e-4, 'rtol': 5e-3})
            self.check_backward_options.update({
                'atol': 2 ** -4, 'rtol': 2 ** -4})
            self.check_double_backward_options.update({
                'atol': 2 ** -4, 'rtol': 2 ** -4})

    def before_test(self, test_name):
        self.backend_config.autotune = self.autotune

    def generate_inputs(self):
        W = numpy.random.normal(
            0, self.W_scale, self.W_shape).astype(self.W_dtype)
        x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.x_dtype)
        if self.nobias:
            return x, W
        else:
            b = numpy.random.uniform(
                -1, 1, self.out_channels).astype(self.x_dtype)
            return x, W, b

    def forward_expected(self, inputs):
        if self.nobias:
            x, W = inputs
            b = None
        else:
            x, W, b = inputs
        y_expected = F.convolution_nd(
            x, W, b, stride=self.stride, pad=self.pad,
            cover_all=self.cover_all, dilate=self.dilate,
            groups=self.groups)
        return y_expected.array,

    def forward(self, inputs, device):
        if self.nobias:
            x, W = inputs
            b = None
        else:
            x, W, b = inputs
        y = F.convolution_nd(
            x, W, b, stride=self.stride, pad=self.pad,
            cover_all=self.cover_all, dilate=self.dilate,
            groups=self.groups)
        return y,

    def check_forward_consistency(self, backend_config):
        inputs = self.generate_inputs()
        if self.nobias:
            x, W = inputs
            b = None
        else:
            x, W, b = inputs
        x_cpu = chainer.Variable(x)
        W_cpu = chainer.Variable(W)
        b_cpu = None if b is None else chainer.Variable(b)
        y_cpu = F.convolution_nd(
            x_cpu, W_cpu, b_cpu, stride=self.stride, pad=self.pad,
            cover_all=self.cover_all, dilate=self.dilate,
            groups=self.groups)

        x = backend_config.get_array(x)
        W = backend_config.get_array(W)
        if self.nobias:
            b = None
        else:
            b = backend_config.get_array(b)
        with backend_config:
            y_gpu = F.convolution_nd(
                x, W, b, stride=self.stride, pad=self.pad,
                cover_all=self.cover_all, dilate=self.dilate,
                groups=self.groups)

        testing.assert_allclose(
            y_cpu.array, y_gpu.array, **self.check_forward_options)

    def test_consistency_forward(self, backend_config):
        if backend_config.use_cuda or backend_config.use_chainerx:
            self.check_forward_consistency(backend_config)

    def check_forward_consistency_regression(self, backend_config):
        inputs = self.generate_inputs()
        if self.nobias:
            x, W = inputs
            b = None
        else:
            x, W, b = inputs
        x = chainer.Variable(backend_config.get_array(x))
        W = chainer.Variable(backend_config.get_array(W))
        if b is not None:
            b = chainer.Variable(backend_config.get_array(b))

        with chainer.using_config('use_cudnn', 'never'):
            y_nd = F.convolution_nd(
                x, W, b, stride=self.stride, pad=self.pad,
                cover_all=self.cover_all, dilate=self.dilate,
                groups=self.groups)
            y_2d = F.convolution_2d(
                x, W, b, stride=self.stride, pad=self.pad,
                cover_all=self.cover_all, dilate=self.dilate,
                groups=self.groups)

        testing.assert_allclose(
            y_nd.array, y_2d.array, **self.check_forward_options)

    def test_consistency_regression_forward(self, backend_config):
        # Regression test to convolution_2d.
        if len(self.dims) == 2:
            self.check_forward_consistency_regression(backend_config)


@testing.parameterize(*testing.product({
    'dims': [(10,), (10, 8), (10, 8, 6)],
    'use_cudnn': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestConvolutionNDCudnnCall(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 2
        ndim = len(self.dims)
        ksize = (3,) * ndim
        self.stride = (2,) * ndim
        self.pad = (1,) * ndim
        x_shape = (2, 3) + self.dims
        self.x = cuda.cupy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        W_scale = numpy.sqrt(1. / functools.reduce(mul, ksize, in_channels))
        W_shape = (out_channels, in_channels) + ksize
        self.W = cuda.cupy.random.normal(
            0, W_scale, W_shape).astype(self.dtype)
        gy_shape = (2, 2) + tuple(
            conv.get_conv_outsize(d, k, s, p) for (d, k, s, p) in zip(
                self.dims, ksize, self.stride, self.pad))
        self.gy = cuda.cupy.random.uniform(-1, 1, gy_shape).astype(self.dtype)
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.expect = chainer.should_use_cudnn('>=auto') and ndim > 1

    def forward(self):
        x = chainer.Variable(cuda.to_gpu(self.x))
        W = chainer.Variable(cuda.to_gpu(self.W))
        return F.convolution_nd(
            x, W, None, stride=self.stride, pad=self.pad)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with testing.patch('cupy.cudnn.convolution_forward') as func:
                self.forward()
                self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            y.grad = self.gy
            name = 'cupy.cudnn.convolution_backward_data'
            with testing.patch(name) as func:
                y.backward()
                self.assertEqual(func.called, self.expect)


class TestConvolutionNDarraySupplied(unittest.TestCase):

    def setUp(self):
        N = 2
        in_channels = 3
        out_channels = 2
        dtype = numpy.float32

        x_shape = (N, in_channels, 3, 3, 3)
        self.x_data = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        W_shape = (out_channels, in_channels, 1, 1, 1)
        self.W_data = numpy.random.uniform(-1, 1, W_shape).astype(dtype)
        self.b_data = numpy.random.uniform(-1, 1, out_channels).astype(dtype)

    def check_array_supplied(self, x_ary, W_ary, b_ary):
        y_ary = F.convolution_nd(x_ary, W_ary, b_ary)

        x_var = chainer.Variable(x_ary)
        W_var = chainer.Variable(W_ary)
        b_var = chainer.Variable(b_ary)
        y_var = F.convolution_nd(x_var, W_var, b_var)

        testing.assert_allclose(y_ary.data, y_var.data)

    def test_array_supplied_cpu(self):
        self.check_array_supplied(self.x_data, self.W_data, self.b_data)

    @attr.gpu
    def test_array_supplied_gpu(self):
        self.check_array_supplied(cuda.to_gpu(self.x_data),
                                  cuda.to_gpu(self.W_data),
                                  cuda.to_gpu(self.b_data))


class TestConvolutionNDBackwardNoncontiguousGradOutputs(unittest.TestCase):
    # NumPy raises an error when the inputs of dot operation are not
    # contiguous. This test ensures this issue is correctly handled.
    # (https://github.com/chainer/chainer/issues/2744)

    # This test depdends on that backward() of F.sum generates
    # a non-contiguous array.

    def test_1(self):
        n_batches = 2
        in_channels = 3
        out_channels = 1  # important
        x_shape = (n_batches, in_channels, 4)
        w_shape = (out_channels, in_channels, 3)
        x = numpy.ones(x_shape, numpy.float32)
        w = numpy.ones(w_shape, numpy.float32)
        y = F.convolution_nd(chainer.Variable(x), w)
        z = F.sum(y)
        z.backward()

    def test_2(self):
        n_batches = 2
        in_channels = 3
        out_channels = 1  # important
        x_shape = (n_batches, in_channels, 4)
        w_shape = (out_channels, in_channels, 3)
        x = numpy.ones(x_shape, numpy.float32)
        w = numpy.ones(w_shape, numpy.float32)
        y = F.convolution_nd(x, chainer.Variable(w))
        z = F.sum(y)
        z.backward()


class TestConvolutionNDWrappers(unittest.TestCase):

    def _get_data(self, ndim):
        in_channels = 3
        out_channels = 2
        dtype = numpy.float32

        x_shape = (2, in_channels) + (3,) * ndim
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        W_shape = (out_channels, in_channels) + (1,) * ndim
        W = numpy.random.uniform(-1, 1, W_shape).astype(dtype)
        b = numpy.random.uniform(-1, 1, out_channels).astype(dtype)

        return x, W, b

    def test_conv1d(self):
        (x, W, b) = self._get_data(1)
        testing.assert_allclose(
            F.convolution_nd(x, W, b).data, F.convolution_1d(x, W, b).data)

    def test_conv1d_invalid(self):
        (x, W, b) = self._get_data(2)
        with self.assertRaises(ValueError):
            F.convolution_1d(x, W, b)

    def test_conv3d(self):
        (x, W, b) = self._get_data(3)
        testing.assert_allclose(
            F.convolution_nd(x, W, b).data, F.convolution_3d(x, W, b).data)

    def test_conv3d_invalid(self):
        (x, W, b) = self._get_data(2)
        with self.assertRaises(ValueError):
            F.convolution_3d(x, W, b)


testing.run_module(__name__, __file__)
