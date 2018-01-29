import unittest

import functools
import mock
import numpy
from operator import mul

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import conv


@testing.parameterize(*(testing.product({
    'dims': [(5,), (4, 3), (3, 4, 3)],
    'cover_all': [True, False],
    'c_contiguous': [True],
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
    'autotune': [True, False],
}) + testing.product({
    'dims': [(4,)],
    'cover_all': [False],
    'c_contiguous': [False],
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'autotune': [False],
})))
class TestConvolutionND(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 2
        ndim = len(self.dims)
        ksize = (3,) * ndim
        self.stride = (2,) * ndim
        self.pad = (1,) * ndim

        W_scale = numpy.sqrt(1. / functools.reduce(mul, ksize, in_channels))
        W_shape = (out_channels, in_channels) + ksize
        self.W = numpy.random.normal(0, W_scale, W_shape).astype(self.W_dtype)
        self.b = numpy.random.uniform(-1, 1, out_channels).astype(self.x_dtype)

        x_shape = (2, 3) + self.dims
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.x_dtype)
        gy_shape = (2, 2) + tuple(
            conv.get_conv_outsize(d, k, s, p, cover_all=self.cover_all)
            for (d, k, s, p) in zip(self.dims, ksize, self.stride, self.pad))
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(self.x_dtype)

        self.check_forward_options = {}
        self.check_backward_options = {
            'dtype': numpy.float64, 'atol': 3e-5, 'rtol': 3e-4}
        if self.x_dtype == numpy.float16 or self.W_dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 2 ** -4, 'rtol': 2 ** -4}

        self.ggx = numpy.random.uniform(-1, 1, self.x.shape).astype(
            self.x_dtype)
        self.ggW = numpy.random.uniform(-1, 1, self.W.shape).astype(
            self.W_dtype)
        self.ggb = numpy.random.uniform(-1, 1, self.b.shape).astype(
            self.x_dtype)

    def check_forward_consistency(self, nobias=False, use_cudnn='never'):
        x_cpu = chainer.Variable(self.x)
        W_cpu = chainer.Variable(self.W)
        b_cpu = None if nobias else chainer.Variable(self.b)
        y_cpu = F.convolution_nd(
            x_cpu, W_cpu, b_cpu, stride=self.stride, pad=self.pad,
            cover_all=self.cover_all)

        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        W_gpu = chainer.Variable(cuda.to_gpu(self.W))
        b_gpu = None if nobias else chainer.Variable(cuda.to_gpu(self.b))
        with chainer.using_config('use_cudnn', use_cudnn):
            with chainer.using_config('autotune', self.autotune):
                y_gpu = F.convolution_nd(
                    x_gpu, W_gpu, b_gpu, stride=self.stride, pad=self.pad,
                    cover_all=self.cover_all)

        testing.assert_allclose(
            y_cpu.data, y_gpu.data, **self.check_forward_options)

    @attr.cudnn
    def test_forward_consistency(self):
        self.check_forward_consistency(nobias=False, use_cudnn='always')

    @attr.cudnn
    def test_forward_consistency_nobias(self):
        self.check_forward_consistency(nobias=True, use_cudnn='always')

    @attr.gpu
    def test_forward_consistency_im2col(self):
        self.check_forward_consistency(nobias=False, use_cudnn='never')

    @attr.gpu
    def test_forward_consistency_im2col_nobias(self):
        self.check_forward_consistency(nobias=True, use_cudnn='never')

    def check_forward_consistency_regression(self, nobias=False):
        x = chainer.Variable(self.x)
        W = chainer.Variable(self.W)
        b = None if nobias else chainer.Variable(self.b)

        with chainer.using_config('use_cudnn', 'never'):
            y_nd = F.convolution_nd(
                x, W, b, stride=self.stride, pad=self.pad,
                cover_all=self.cover_all)
            y_2d = F.convolution_2d(
                x, W, b, stride=self.stride, pad=self.pad,
                cover_all=self.cover_all)

        testing.assert_allclose(
            y_nd.data, y_2d.data, **self.check_forward_options)

    def test_forward_consistency_regression(self):
        # Regression test to convolution_2d.
        if len(self.dims) == 2:
            self.check_forward_consistency_regression(nobias=False)

    def test_forward_consistency_regression_nobias(self):
        # Regression test to convolution_2d.
        if len(self.dims) == 2:
            self.check_forward_consistency_regression(nobias=True)

    def check_backward(self, x_data, W_data, b_data, y_grad,
                       use_cudnn='never'):
        xp = cuda.get_array_module(x_data)
        if not self.c_contiguous:
            x_data = xp.asfortranarray(x_data)
            W_data = xp.asfortranarray(W_data)
            y_grad = xp.asfortranarray(y_grad)
            self.assertTrue(x_data.flags.f_contiguous)
            self.assertTrue(W_data.flags.f_contiguous)
            self.assertTrue(y_grad.flags.f_contiguous)
            if b_data is not None:
                b = xp.empty((len(b_data) * 2,), dtype=b_data.dtype)
                b[::2] = b_data
                b_data = b[::2]
                self.assertFalse(b_data.flags.c_contiguous)

        args = (x_data, W_data)
        if b_data is not None:
            args += (b_data,)

        def f(*args):
            return F.convolution_nd(
                *args, stride=self.stride, pad=self.pad,
                cover_all=self.cover_all)

        with chainer.using_config('use_cudnn', use_cudnn):
            with chainer.using_config('autotune', self.autotune):
                gradient_check.check_backward(
                    f, args, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @condition.retry(3)
    def test_backward_cpu_nobias(self):
        self.check_backward(self.x, self.W, None, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy),
                            use_cudnn='always')

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu_nobias(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy),
                            use_cudnn='always')

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_im2col(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy),
                            use_cudnn='never')

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_im2col_nobias(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy),
                            use_cudnn='never')

    def check_double_backward(self, x_data, W_data, b_data, y_grad,
                              x_grad_grad, W_grad_grad, b_grad_grad,
                              use_cudnn='always'):
        xp = cuda.get_array_module(x_data)

        if not self.c_contiguous:
            x_data = xp.asfortranarray(x_data)
            W_data = xp.asfortranarray(W_data)
            y_grad = xp.asfortranarray(y_grad)
            x_grad_grad = xp.asfortranarray(x_grad_grad)
            W_grad_grad = xp.asfortranarray(W_grad_grad)
            self.assertFalse(x_data.flags.c_contiguous)
            self.assertFalse(W_data.flags.c_contiguous)
            self.assertFalse(y_grad.flags.c_contiguous)
            self.assertFalse(x_grad_grad.flags.c_contiguous)
            self.assertFalse(W_grad_grad.flags.c_contiguous)
            if b_data is not None:
                b = xp.empty((len(b_data) * 2,), dtype=self.b.dtype)
                b[::2] = b_data
                b_data = b[::2]
                self.assertFalse(b_data.flags.c_contiguous)

                ggb = xp.empty((len(b_data) * 2,), dtype=self.b.dtype)
                ggb[::2] = b_grad_grad
                b_grad_grad = ggb[::2]
                self.assertFalse(b_grad_grad.flags.c_contiguous)

        args = (x_data, W_data)
        grad_grads = (x_grad_grad, W_grad_grad)
        if b_data is not None:
            args += (b_data,)
            grad_grads += (b_grad_grad,)

        def f(*args):
            y = F.convolution_nd(*args, stride=self.stride, pad=self.pad,
                                 cover_all=self.cover_all)
            return y * y  # make the function nonlinear

        with chainer.using_config('use_cudnn', use_cudnn):
            with chainer.using_config('autotune', self.autotune):
                gradient_check.check_double_backward(
                    f, args, y_grad, grad_grads,
                    dtype='d', atol=5e-3, rtol=5e-2)

    @condition.retry(3)
    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.W, self.b, self.gy,
                                   self.ggx, self.ggW, self.ggb,
                                   use_cudnn='always')

    @condition.retry(3)
    def test_double_backward_cpu_nobias(self):
        self.check_double_backward(self.x, self.W, None, self.gy,
                                   self.ggx, self.ggW, None,
                                   use_cudnn='always')

    def check_double_backward_gpu(self, bias=True, im2col=False):
        use_cudnn = 'never' if im2col else 'always'
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W),
            cuda.to_gpu(self.b) if bias else None,
            cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx), cuda.to_gpu(self.ggW),
            cuda.to_gpu(self.ggb) if bias else None,
            use_cudnn=use_cudnn)

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_gpu(self):
        self.check_double_backward_gpu()

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_gpu_nobias(self):
        self.check_double_backward_gpu(bias=False)

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_gpu_im2col(self):
        self.check_double_backward_gpu(im2col=True)

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_gpu_im2col_nobias(self):
        self.check_double_backward_gpu(bias=False, im2col=True)


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
            with mock.patch('cupy.cuda.cudnn.convolutionForward') as func:
                self.forward()
                self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            y.grad = self.gy
            name = 'cupy.cuda.cudnn.convolutionBackwardData_v3'
            with mock.patch(name) as func:
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


testing.run_module(__name__, __file__)
