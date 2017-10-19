import mock
import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*(testing.product({
    'c_contiguous': [True, False],
    'cover_all': [True, False],
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
    'cudnn_deterministic': [True, False],
}) + testing.product({
    'c_contiguous': [False],
    'cover_all': [False],
    'cudnn_deterministic': [False],
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
class TestConvolution2DFunction(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 2
        kh, kw = (3, 3)
        self.stride = 2
        self.pad = 1
        self.use_cudnn = 'always'
        self.W = numpy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * in_channels)),
            (out_channels, in_channels, kh, kw)).astype(self.W_dtype)
        self.b = numpy.random.uniform(
            -1, 1, out_channels).astype(self.x_dtype)

        self.x = numpy.random.uniform(
            -1, 1, (2, 3, 4, 3)).astype(self.x_dtype)
        if self.cover_all:
            self.gy = numpy.random.uniform(-1, 1,
                                           (2, 2, 3, 2)).astype(self.x_dtype)
        else:
            self.gy = numpy.random.uniform(
                -1, 1, (2, 2, 2, 2)).astype(self.x_dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.x.shape).astype(
            self.x_dtype)
        self.ggW = numpy.random.uniform(-1, 1, self.W.shape).astype(
            self.W_dtype)
        self.ggb = numpy.random.uniform(-1, 1, self.b.shape).astype(
            self.x_dtype)
        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        self.check_double_backward_options = {'dtype': numpy.float64}
        if self.x_dtype == numpy.float16 or self.W_dtype == numpy.float16:
            self.check_forward_options.update(atol=5e-4, rtol=5e-3)
            self.check_backward_options.update(atol=5e-4, rtol=5e-3)
            self.check_double_backward_options.update(atol=5e-3, rtol=5e-2)

    @attr.gpu
    def test_forward_consistency(self, nobias=False):
        x_cpu = chainer.Variable(self.x)
        W_cpu = chainer.Variable(self.W)
        b_cpu = None if nobias else chainer.Variable(self.b)
        with chainer.using_config('cudnn_deterministic',
                                  self.cudnn_deterministic):
            y_cpu = F.convolution_2d(
                x_cpu, W_cpu, b_cpu, stride=self.stride, pad=self.pad,
                cover_all=self.cover_all)

        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        W_gpu = chainer.Variable(cuda.to_gpu(self.W))
        b_gpu = None if nobias else chainer.Variable(cuda.to_gpu(self.b))
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with chainer.using_config('cudnn_deterministic',
                                      self.cudnn_deterministic):
                y_gpu = F.convolution_2d(
                    x_gpu, W_gpu, b_gpu, stride=self.stride, pad=self.pad,
                    cover_all=self.cover_all)

        testing.assert_allclose(
            y_cpu.data, y_gpu.data.get(), **self.check_forward_options)

    @attr.gpu
    def test_forward_consistency_im2col(self):
        self.use_cudnn = 'never'
        self.test_forward_consistency()

    @attr.gpu
    def test_forward_consistency_im2col_nobias(self):
        self.use_cudnn = 'never'
        self.test_forward_consistency(nobias=True)

    def check_backward(self, x_data, W_data, b_data, y_grad):
        xp = cuda.get_array_module(x_data)

        if not self.c_contiguous:
            x_data = xp.asfortranarray(x_data)
            W_data = xp.asfortranarray(W_data)
            y_grad = xp.asfortranarray(y_grad)
            self.assertFalse(x_data.flags.c_contiguous)
            self.assertFalse(W_data.flags.c_contiguous)
            self.assertFalse(y_grad.flags.c_contiguous)
            if b_data is not None:
                b = xp.empty((len(b_data) * 2,), dtype=self.b.dtype)
                b[::2] = b_data
                b_data = b[::2]
                self.assertFalse(b_data.flags.c_contiguous)

        args = (x_data, W_data)
        if b_data is not None:
            args = args + (b_data,)

        def f(*args):
            return F.convolution_2d(*args, stride=self.stride, pad=self.pad,
                                    cover_all=self.cover_all)

        with chainer.using_config('use_cudnn', self.use_cudnn):
            with chainer.using_config('cudnn_deterministic',
                                      self.cudnn_deterministic):
                gradient_check.check_backward(
                    f, args, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @condition.retry(3)
    def test_backward_cpu_nobias(self):
        self.check_backward(self.x, self.W, None, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_nobias(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_im2col(self):
        self.use_cudnn = 'never'
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_im2col_nobias(self):
        self.use_cudnn = 'never'
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, W_data, b_data, y_grad,
                              x_grad_grad, W_grad_grad, b_grad_grad):
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
            args = args + (b_data,)
            grad_grads = grad_grads + (b_grad_grad,)

        def f(*args):
            y = F.convolution_2d(*args, stride=self.stride, pad=self.pad,
                                 cover_all=self.cover_all)
            return y * y  # make the function nonlinear

        with chainer.using_config('use_cudnn', self.use_cudnn):
            with chainer.using_config('cudnn_deterministic',
                                      self.cudnn_deterministic):
                gradient_check.check_double_backward(
                    f, args, y_grad, grad_grads,
                    **self.check_double_backward_options)

    @condition.retry(3)
    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.W, self.b, self.gy,
                                   self.ggx, self.ggW, self.ggb)

    @condition.retry(3)
    def test_double_backward_cpu_nobias(self):
        self.check_double_backward(self.x, self.W, None, self.gy,
                                   self.ggx, self.ggW, None)

    def check_double_backward_gpu(self, bias=True, im2col=False):
        if im2col:
            self.use_cudnn = 'never'
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W),
            cuda.to_gpu(self.b) if bias else None,
            cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx), cuda.to_gpu(self.ggW),
            cuda.to_gpu(self.ggb) if bias else None)

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
    'use_cudnn': ['always', 'auto', 'never'],
    'cudnn_deterministic': [False, True],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestConvolution2DCudnnCall(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 2
        kh, kw = (3, 3)
        self.stride = 2
        self.pad = 1
        self.x = cuda.cupy.random.uniform(
            -1, 1, (2, 3, 4, 3)).astype(self.dtype)
        self.W = cuda.cupy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * in_channels)),
            (out_channels, in_channels, kh, kw)).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(
            -1, 1, (2, 2, 2, 2)).astype(self.dtype)
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.should_call_cudnn = chainer.should_use_cudnn('>=auto')

    def forward(self):
        x = chainer.Variable(self.x)
        W = chainer.Variable(self.W)
        return F.convolution_2d(x, W, None, stride=self.stride, pad=self.pad)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with chainer.using_config('cudnn_deterministic',
                                      self.cudnn_deterministic):
                with mock.patch('cupy.cudnn.cudnn.convolutionForward') as func:
                    self.forward()
                    self.assertEqual(func.called, self.should_call_cudnn)

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with chainer.using_config('cudnn_deterministic',
                                      self.cudnn_deterministic):
                y = self.forward()
                y.grad = self.gy
                name = 'cupy.cudnn.cudnn.convolutionBackwardData_v3'
                with mock.patch(name) as func:
                    y.backward()
                self.assertEqual(func.called, self.should_call_cudnn)


@testing.parameterize(*testing.product({
    'c_contiguous': [True, False],
    'nobias': [True, False],
}))
@attr.gpu
@attr.cudnn
class TestConvolution2DFunctionCudnnDeterministic(unittest.TestCase):

    def setUp(self):
        self.stride = 2
        self.pad = 1
        batch_sz = 2
        in_channels = 64
        out_channels = 64
        kh, kw = (3, 3)
        in_h, in_w = (32, 128)
        out_h, out_w = (16, 64)
        # should be same types for cudnn test
        x_dtype = numpy.float32
        W_dtype = numpy.float32
        self.W = numpy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * in_channels)),
            (out_channels, in_channels, kh, kw)).astype(W_dtype)
        self.b = numpy.random.uniform(-1, 1, out_channels).astype(x_dtype)
        self.x = numpy.random.uniform(
            -1, 1, (batch_sz, in_channels, in_h, in_w)).astype(x_dtype)
        self.gy = numpy.random.uniform(
            -1, 1, (batch_sz, out_channels, out_h, out_w)).astype(x_dtype)

    def test_called(self):
        with mock.patch(
            'chainer.functions.connection.convolution_2d.libcudnn',
            autospec=True
        ) as mlibcudnn_conv, mock.patch(
            'chainer.functions.connection.deconvolution_2d.libcudnn',
            autospec=True
        ) as mlibcudnn_deconv:

            # cuDNN version >= v3 supports `cudnn_deterministic` option
            x, W, b, y = self._run()

            # in Convolution2DFunction.backward_gpu()
            self.assertFalse(
                mlibcudnn_conv.getConvolutionBackwardFilterAlgorithm.called)
            self.assertEqual(
                mlibcudnn_conv.convolutionBackwardFilter_v3.call_count, 1)
            self.assertFalse(
                mlibcudnn_deconv.getConvolutionBackwardDataAlgorithm.called)
            self.assertEqual(
                mlibcudnn_deconv.convolutionBackwardData_v3.call_count, 1)

    def test_cudnn_deterministic(self):
        x1, W1, b1, y1 = self._run()
        x2, W2, b2, y2 = self._run()

        cuda.cupy.testing.assert_array_equal(x1.grad, x2.grad)
        cuda.cupy.testing.assert_array_equal(y1.data, y2.data)
        cuda.cupy.testing.assert_array_equal(W1.grad, W2.grad)

    def _contiguous(self, x_data, W_data, b_data, gy_data):
        if not self.c_contiguous:
            x_data = numpy.asfortranarray(x_data)
            W_data = numpy.asfortranarray(W_data)
            gy_data = numpy.asfortranarray(gy_data)
            self.assertFalse(x_data.flags.c_contiguous)
            self.assertFalse(W_data.flags.c_contiguous)
            self.assertFalse(gy_data.flags.c_contiguous)
            b = numpy.empty((len(b_data) * 2,), dtype=self.b.dtype)
            b[::2] = b_data
            b_data = b[::2]
            self.assertFalse(b_data.flags.c_contiguous)
        return x_data, W_data, b_data, gy_data

    def _run(self):
        with chainer.using_config('use_cudnn', 'always'):
            with chainer.using_config('cudnn_deterministic', True):
                # verify data continuity and move to gpu
                x_data, W_data, b_data, gy_data = \
                    tuple(cuda.to_gpu(data) for data in self._contiguous(
                        self.x, self.W, self.b, self.gy))
                x, W, b, y = self._run_forward(x_data, W_data, b_data)

                y.grad = gy_data
                y.backward()
                return x, W, b, y

    def _run_forward(self, x_data, W_data, b_data):
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        b = None if self.nobias else chainer.Variable(b_data)
        y = F.convolution_2d(x, W, b, stride=self.stride, pad=self.pad,
                             cover_all=False)
        return x, W, b, y


class TestConvolution2DBackwardNoncontiguousGradOutputs(unittest.TestCase):
    # NumPy raises an error when the inputs of dot operation are not
    # contiguous. This test ensures this issue is correctly handled.
    # (https://github.com/chainer/chainer/issues/2744)

    # This test depdends on that backward() of F.sum generates
    # a non-contiguous array.

    def test_1(self):
        n_batches = 2
        in_channels = 3
        out_channels = 1  # important
        x_shape = (n_batches, in_channels, 10, 10)
        w_shape = (out_channels, in_channels, 3, 3)
        x = numpy.ones(x_shape, numpy.float32)
        w = numpy.ones(w_shape, numpy.float32)
        y = F.convolution_2d(x, chainer.Variable(w))
        z = F.sum(y)
        z.backward()


testing.run_module(__name__, __file__)
