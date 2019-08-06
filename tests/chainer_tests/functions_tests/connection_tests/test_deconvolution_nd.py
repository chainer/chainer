import unittest

import functools
import numpy
from operator import mul

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import testing
from chainer.testing import attr
from chainer.testing import parameterize
from chainer.utils import conv
from chainer.utils import type_check


@parameterize(*testing.product({
    'dims': [(4, 3, 2), (2,)],
    'dilate': [1, 2],
    'groups': [1, 2],
    'nobias': [False],
    'test_outsize': [False],
    'contiguous': ['C'],
    'b_dtype': [numpy.float32],
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
}) + testing.product({
    'dims': [(3, 2)],
    'dilate': [1, 2],
    'groups': [1],
    'nobias': [False],
    'test_outsize': [False],
    'contiguous': ['C'],
    'b_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
}) + testing.product({
    'dims': [(3, 2)],
    'dilate': [1, 2],
    'groups': [1],
    'nobias': [True, False],
    'test_outsize': [True, False],
    'contiguous': ['C', None],
    'b_dtype': [numpy.float32],
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
}))
@testing.inject_backend_tests(
    None,
    # ChainerX tests
    testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
    # CPU tests
    + testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + testing.product([
        [{'use_cuda': True}],
        # Without cuDNN
        testing.product({
            'use_cudnn': ['never'],
        })
        # With cuDNN
        + testing.product({
            'use_cudnn': ['always'],
            'autotune': [True, False],
        })]))
class TestDeconvolutionND(testing.FunctionTestCase):

    def setUp(self):
        self.N = 2
        self.in_channels = 4
        self.out_channels = 2
        self.ndim = len(self.dims)
        self.ksize = (3,) * self.ndim
        self.stride = (2,) * self.ndim
        self.pad = (1,) * self.ndim
        self.dilate = (self.dilate,) * self.ndim

        self.W_scale = numpy.sqrt(1. / functools.reduce(mul, self.ksize,
                                                        self.in_channels))
        self.W_shape = (self.in_channels,
                        self.out_channels // self.groups) + self.ksize

        outs = tuple(
            conv.get_deconv_outsize(d, k, s, p, d=di)
            for (d, k, s, p, di)
            in zip(self.dims, self.ksize, self.stride, self.pad, self.dilate))
        self.outsize = outs if self.test_outsize else None
        self.x_shape = (self.N, self.in_channels) + self.dims
        self.gy_shape = (self.N, self.out_channels) + outs

        self.check_backward_options.update({'atol': 3e-5, 'rtol': 3e-4})
        self.check_double_backward_options.update({'atol': 5e-3, 'rtol': 5e-2})
        if (self.x_dtype == numpy.float16 or self.W_dtype == numpy.float16
                or self.b_dtype == numpy.float16):
            self.check_forward_options.update({'atol': 5e-3, 'rtol': 5e-3})
            self.check_backward_options.update({
                'atol': 2 ** -4, 'rtol': 2 ** -4})
            self.check_double_backward_options.update({
                'atol': 2 ** -4, 'rtol': 2 ** -4})

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
        """
        Current forward_expected implementation depends on
        F.deconvolution_nd itself and thus it's only capable
        of checking consistency between backends, not absolute
        correctness of computations
        """
        if self.nobias:
            x, W = inputs
            b = None
        else:
            x, W, b = inputs
        y_expected = F.deconvolution_nd(
            x, W, b, stride=self.stride, pad=self.pad,
            outsize=self.outsize, dilate=self.dilate,
            groups=self.groups)
        return y_expected.array,

    def forward(self, inputs, device):
        if self.nobias:
            x, W = inputs
            b = None
        else:
            x, W, b = inputs
        y = F.deconvolution_nd(
            x, W, b, stride=self.stride, pad=self.pad,
            outsize=self.outsize, dilate=self.dilate,
            groups=self.groups)
        return y,

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

        use_cudnn = backend_config.use_cudnn

        with chainer.using_config('use_cudnn', use_cudnn):
            y_nd = F.deconvolution_nd(x, W, b, stride=self.stride,
                                      pad=self.pad, outsize=self.outsize,
                                      dilate=self.dilate)
            y_2d = F.deconvolution_2d(x, W, b, stride=self.stride,
                                      pad=self.pad, outsize=self.outsize,
                                      dilate=self.dilate)

        testing.assert_allclose(
            y_nd.array, y_2d.array, **self.check_forward_options)

    def test_consistency_regression_forward(self, backend_config):
        # Regression test to convolution_2d.
        if len(self.dims) == 2:
            self.check_forward_consistency_regression(backend_config)


@testing.parameterize(*testing.product({
    'dims': [(5, 4, 3), (4, 3), (3,)],
    'use_cudnn': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestDeconvolutionNDCudnnCall(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 2
        ndim = len(self.dims)
        ksize = (3,) * ndim
        stride = (1,) * ndim
        pad = (1,) * ndim

        W_scale = numpy.sqrt(1. / functools.reduce(mul, ksize, in_channels))
        W_shape = (in_channels, out_channels) + ksize
        self.W = cuda.cupy.random.normal(
            0, W_scale, W_shape).astype(self.dtype)
        outs = tuple(
            conv.get_deconv_outsize(d, k, s, p)
            for (d, k, s, p) in zip(self.dims, ksize, stride, pad))
        x_shape = (2, in_channels) + self.dims
        self.x = cuda.cupy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        gy_shape = (2, out_channels) + outs
        self.gy = cuda.cupy.random.uniform(-1, 1, gy_shape).astype(self.dtype)
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.expected = chainer.should_use_cudnn('>=auto') and ndim > 1

    def forward(self):
        x = chainer.Variable(self.x)
        W = chainer.Variable(self.W)
        return F.deconvolution_nd(x, W, None, stride=1, pad=1)

    def test_call_cudnn_forward(self):
        name = 'cupy.cudnn.convolution_backward_data'
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with testing.patch(name) as func:
                self.forward()
                self.assertEqual(func.called, self.expected)

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            y.grad = self.gy
            with testing.patch('cupy.cudnn.convolution_forward') as func:
                y.backward()
                self.assertEqual(func.called, self.expected)


class TestDeconvolutionNDarraySupplied(unittest.TestCase):

    def setUp(self):
        N = 2
        in_channels = 3
        out_channels = 2
        dtype = numpy.float32

        x_shape = (N, in_channels, 3, 3, 3)
        self.x_data = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        W_shape = (in_channels, out_channels, 1, 1, 1)
        self.W_data = numpy.random.uniform(-1, 1, W_shape).astype(dtype)
        self.b_data = numpy.random.uniform(-1, 1, out_channels).astype(dtype)

    def check_array_supplied(self, x_ary, W_ary, b_ary):
        y_ary = F.deconvolution_nd(x_ary, W_ary, b_ary)

        x_var = chainer.Variable(x_ary)
        W_var = chainer.Variable(W_ary)
        b_var = chainer.Variable(b_ary)
        y_var = F.deconvolution_nd(x_var, W_var, b_var)

        testing.assert_allclose(y_ary.data, y_var.data)

    def test_array_supplied_cpu(self):
        self.check_array_supplied(self.x_data, self.W_data, self.b_data)

    @attr.gpu
    def test_array_supplied_gpu(self):
        self.check_array_supplied(cuda.to_gpu(self.x_data),
                                  cuda.to_gpu(self.W_data),
                                  cuda.to_gpu(self.b_data))


class TestDeconvolutionNDTypeCheck(unittest.TestCase):

    def test_number_of_inputs(self):
        # Too few inputs
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            F.connection.deconvolution_nd.DeconvolutionND(1).apply((x,))

        # Too much inputs
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.float32)
        b = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            F.connection.deconvolution_nd.DeconvolutionND(1).apply(
                (x, W, b, x))

    def test_data_and_weight(self):
        # dtype of data
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.int32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            F.deconvolution_nd(x, W)

        # dtype of weight
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.int32)
        with self.assertRaises(type_check.InvalidType):
            F.deconvolution_nd(x, W)

        # ndim of weight
        x = numpy.random.uniform(-1, 1, (2, 3, 4, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            F.deconvolution_nd(x, W)

        # shapes of data and weight
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (2, 2, 2)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            F.deconvolution_nd(x, W)

    def test_supplied_outsize(self):
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.float32)
        outsize = (10,)
        with self.assertRaises(type_check.InvalidType):
            F.deconvolution_nd(x, W, outsize=outsize)

    def test_bias(self):
        # dtype
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.float32)
        b = numpy.random.uniform(-1, 1, (2,)).astype(numpy.int32)
        with self.assertRaises(type_check.InvalidType):
            F.deconvolution_nd(x, W, b=b)

        # ndim
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.float32)
        b = numpy.random.uniform(-1, 1, (2, 2)).astype(numpy.float32)
        with self.assertRaises(type_check.InvalidType):
            F.deconvolution_nd(x, W, b=b)

    def test_estimated_outsize(self):
        x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(numpy.float32)
        W = numpy.random.uniform(-1, 1, (3, 2, 2)).astype(numpy.float32)
        stride = 1
        pad = 10
        with self.assertRaises(AssertionError):
            F.deconvolution_nd(x, W, stride=stride, pad=pad)


class TestDeconvolutionNDWrappers(unittest.TestCase):

    def _get_data(self, ndim):
        in_channels = 3
        out_channels = 2
        dtype = numpy.float32

        x_shape = (2, in_channels) + (3,) * ndim
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        W_shape = (in_channels, out_channels) + (1,) * ndim
        W = numpy.random.uniform(-1, 1, W_shape).astype(dtype)
        b = numpy.random.uniform(-1, 1, out_channels).astype(dtype)

        return x, W, b

    def test_deconv1d(self):
        (x, W, b) = self._get_data(1)
        testing.assert_allclose(
            F.deconvolution_nd(x, W, b).data, F.deconvolution_1d(x, W, b).data)

    def test_deconv1d_invalid(self):
        (x, W, b) = self._get_data(2)
        with self.assertRaises(ValueError):
            F.deconvolution_1d(x, W, b)

    def test_deconv3d(self):
        (x, W, b) = self._get_data(3)
        testing.assert_allclose(
            F.deconvolution_nd(x, W, b).data, F.deconvolution_3d(x, W, b).data)

    def test_deconv3d_invalid(self):
        (x, W, b) = self._get_data(2)
        with self.assertRaises(ValueError):
            F.deconvolution_3d(x, W, b)


testing.run_module(__name__, __file__)
