import mock
import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend
from chainer.testing import condition


@testing.parameterize(*(testing.product({
    'c_contiguous': [True, False],
    'cover_all': [True, False],
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
    'dilate': [1],
    'group': [1, 2],
    'nobias': [True, False],
}) + testing.product({
    'c_contiguous': [False],
    'cover_all': [False],
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'dilate': [1],
    'group': [1, 2],
    'nobias': [True, False],
})))
@backend.inject_backend_tests(
    ['test_forward', 'test_backward', 'test_double_backward'],
    # CPU tests
    testing.product({
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
            'cudnn_deterministic': [True, False],
            'autotune': [True, False],
        })]))
class TestConvolution2DFunction(unittest.TestCase):

    def setUp(self):
        batches = 2
        in_channels_a_group = 3
        out_channels_a_group = 2
        in_channels = in_channels_a_group * self.group
        out_channels = out_channels_a_group * self.group
        kh, kw = (3, 3)
        self.stride = 2
        self.pad = (int(kh / 2) * self.dilate, int(kw / 2) * self.dilate)
        W = numpy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * in_channels_a_group)),
            (out_channels, in_channels_a_group, kh, kw)).astype(self.W_dtype)

        if self.nobias:
            b = None
        else:
            b = numpy.random.uniform(
                -1, 1, out_channels).astype(self.x_dtype)

        x = numpy.random.uniform(
            -1, 1, (batches, in_channels, 4, 3)).astype(self.x_dtype)
        if self.cover_all:
            gy = numpy.random.uniform(
                -1, 1, (batches, out_channels, 3, 2)).astype(self.x_dtype)
        else:
            gy = numpy.random.uniform(
                -1, 1, (batches, out_channels, 2, 2)).astype(self.x_dtype)
        ggx = numpy.random.uniform(-1, 1, x.shape).astype(
            self.x_dtype)
        ggW = numpy.random.uniform(-1, 1, W.shape).astype(
            self.W_dtype)
        ggb = None if b is None else numpy.random.uniform(
            -1, 1, b.shape).astype(self.x_dtype)

        self.inputs = [x, W, b]
        self.grad_outputs = [gy]
        self.grad_grad_inputs = [ggx, ggW, ggb]

    def forward_cpu(self, inputs):
        x, W, b = inputs
        x_cpu = chainer.Variable(x)
        W_cpu = chainer.Variable(W)
        b_cpu = None if b is None else chainer.Variable(b)
        with chainer.using_config('use_ideep', 'never'):
            y_cpu = F.convolution_2d(
                x_cpu, W_cpu, b_cpu, stride=self.stride, pad=self.pad,
                cover_all=self.cover_all, dilate=self.dilate, group=self.group)
        return y_cpu,

    def check_forward(self, inputs, backend_config):
        y_expected, = self.forward_cpu(inputs)

        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)

        x, W, b = inputs
        x = chainer.Variable(x)
        W = chainer.Variable(W)
        b = None if b is None else chainer.Variable(b)
        with backend_config:
            y_actual = F.convolution_2d(
                x, W, b, stride=self.stride, pad=self.pad,
                cover_all=self.cover_all, dilate=self.dilate, group=self.group)

        testing.assert_allclose(
            y_expected.data, y_actual.data, atol=5e-4, rtol=5e-3)

    def test_forward(self, backend_config):
        self.check_forward(self.inputs, backend_config)

    def check_backward(self, inputs, grad_outputs, backend_config):
        xp = backend_config.xp
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)

        x_data, W_data, b_data = inputs
        y_grad, = grad_outputs

        if not self.c_contiguous:
            x_data = xp.asfortranarray(x_data)
            W_data = xp.asfortranarray(W_data)
            y_grad = xp.asfortranarray(y_grad)
            assert not x_data.flags.c_contiguous
            assert not W_data.flags.c_contiguous
            assert not y_grad.flags.c_contiguous
            if b_data is not None:
                b = xp.empty((len(b_data) * 2,), dtype=b_data.dtype)
                b[::2] = b_data
                b_data = b[::2]
                assert not b_data.flags.c_contiguous

        args = (x_data, W_data)
        if b_data is not None:
            args = args + (b_data,)

        def f(*args):
            return F.convolution_2d(*args, stride=self.stride, pad=self.pad,
                                    cover_all=self.cover_all,
                                    dilate=self.dilate, group=self.group)

        with backend_config:
            gradient_check.check_backward(
                f, args, y_grad, dtype='d', atol=5e-4, rtol=5e-3)

    @condition.retry(3)
    def test_backward(self, backend_config):
        self.check_backward(self.inputs, self.grad_outputs, backend_config)

    def check_double_backward(
            self, inputs, grad_outputs, grad_grad_inputs, backend_config):
        xp = backend_config.xp

        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)
            grad_grad_inputs = cuda.to_gpu(grad_grad_inputs)

        x_data, W_data, b_data = inputs
        y_grad, = grad_outputs
        x_grad_grad, W_grad_grad, b_grad_grad = grad_grad_inputs

        if not self.c_contiguous:
            x_data = xp.asfortranarray(x_data)
            W_data = xp.asfortranarray(W_data)
            y_grad = xp.asfortranarray(y_grad)
            x_grad_grad = xp.asfortranarray(x_grad_grad)
            W_grad_grad = xp.asfortranarray(W_grad_grad)
            assert not x_data.flags.c_contiguous
            assert not W_data.flags.c_contiguous
            assert not y_grad.flags.c_contiguous
            assert not x_grad_grad.flags.c_contiguous
            assert not W_grad_grad.flags.c_contiguous
            if b_data is not None:
                b = xp.empty((len(b_data) * 2,), dtype=b_data.dtype)
                b[::2] = b_data
                b_data = b[::2]
                assert not b_data.flags.c_contiguous

                ggb = xp.empty((len(b_data) * 2,), dtype=b_data.dtype)
                ggb[::2] = b_grad_grad
                b_grad_grad = ggb[::2]
                assert not b_grad_grad.flags.c_contiguous

        args = (x_data, W_data)
        grad_grads = (x_grad_grad, W_grad_grad)
        if b_data is not None:
            args = args + (b_data,)
            grad_grads = grad_grads + (b_grad_grad,)

        def f(*args):
            y = F.convolution_2d(*args, stride=self.stride, pad=self.pad,
                                 cover_all=self.cover_all, dilate=self.dilate,
                                 group=self.group)
            return y * y  # make the function nonlinear

        with backend_config:
            gradient_check.check_double_backward(
                f, args, y_grad, grad_grads,
                dtype='d', atol=5e-3, rtol=5e-2)

    @condition.retry(3)
    def test_double_backward(self, backend_config):
        self.check_double_backward(
            self.inputs, self.grad_outputs, self.grad_grad_inputs,
            backend_config)


@testing.parameterize(*(testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
    'cudnn_deterministic': [False, True],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'dilate': [1],
    'group': [1, 2],
}) + testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
    'cudnn_deterministic': [False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'dilate': [2],
    'group': [1, 2],
})))
@attr.cudnn
class TestConvolution2DCudnnCall(unittest.TestCase):

    def setUp(self):
        batches = 2
        in_channels_a_group = 3
        out_channels_a_group = 2
        in_channels = in_channels_a_group * self.group
        out_channels = out_channels_a_group * self.group
        kh, kw = (3, 3)
        self.stride = 2
        self.pad = (int(kh / 2) * self.dilate, int(kw / 2) * self.dilate)
        self.x = cuda.cupy.random.uniform(
            -1, 1, (batches, in_channels, 4, 3)).astype(self.dtype)
        self.W = cuda.cupy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * in_channels_a_group)),
            (out_channels, in_channels_a_group, kh, kw)).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(
            -1, 1, (batches, out_channels, 2, 2)).astype(self.dtype)
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.should_call_cudnn = chainer.should_use_cudnn('>=auto')
            if self.dilate > 1 and cuda.cuda.cudnn.getVersion() < 6000:
                self.should_call_cudnn = False
            if self.group > 1 and cuda.cuda.cudnn.getVersion() < 7000:
                self.should_call_cudnn = False
            self.can_use_tensor_core = True
            if self.dilate > 1:
                self.can_use_tensor_core = False

    def forward(self):
        x = chainer.Variable(self.x)
        W = chainer.Variable(self.W)
        return F.convolution_2d(x, W, None, stride=self.stride, pad=self.pad,
                                dilate=self.dilate, group=self.group)

    def test_call_cudnn_forward(self):
        name = 'cupy.cuda.cudnn.convolutionForward'
        name2 = 'chainer.functions.connection.convolution_2d' \
            '.Convolution2DFunction._tensor_core_adjust_algo'
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with chainer.using_config('cudnn_deterministic',
                                      self.cudnn_deterministic):
                with mock.patch(
                    name
                ) as func, mock.patch(
                    name2
                ) as tensor_core_adjust_algo:
                    self.forward()
                    self.assertEqual(func.called, self.should_call_cudnn)
                    if not self.can_use_tensor_core:
                        self.assertEqual(tensor_core_adjust_algo.called, False)

    def test_call_cudnn_backward(self):
        name = 'cupy.cuda.cudnn.convolutionBackwardData_v3'
        name2 = 'chainer.functions.connection.convolution_2d' \
            '.Convolution2DGradW._tensor_core_adjust_algo'
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with chainer.using_config('cudnn_deterministic',
                                      self.cudnn_deterministic):
                y = self.forward()
                y.grad = self.gy
                with mock.patch(
                    name
                ) as func, mock.patch(
                    name2
                ) as tensor_core_adjust_algo:
                    y.backward()
                    self.assertEqual(func.called, self.should_call_cudnn)
                    if not self.can_use_tensor_core:
                        self.assertEqual(tensor_core_adjust_algo.called, False)


@testing.parameterize(*testing.product({
    'c_contiguous': [True, False],
    'nobias': [True, False],
    'group': [1, 2],
}))
@attr.gpu
@attr.cudnn
class TestConvolution2DFunctionCudnnDeterministic(unittest.TestCase):

    def setUp(self):
        self.stride = 2
        self.pad = 1
        batch_sz = 2
        in_channels_a_group = 64
        out_channels_a_group = 64
        in_channels = in_channels_a_group * self.group
        out_channels = out_channels_a_group * self.group
        kh, kw = (3, 3)
        in_h, in_w = (32, 128)
        out_h, out_w = (16, 64)
        # should be same types for cudnn test
        x_dtype = numpy.float32
        W_dtype = numpy.float32
        self.W = numpy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * in_channels_a_group)),
            (out_channels, in_channels_a_group, kh, kw)).astype(W_dtype)
        self.b = numpy.random.uniform(-1, 1, out_channels).astype(x_dtype)
        self.x = numpy.random.uniform(
            -1, 1, (batch_sz, in_channels, in_h, in_w)).astype(x_dtype)
        self.gy = numpy.random.uniform(
            -1, 1, (batch_sz, out_channels, out_h, out_w)).astype(x_dtype)
        self.should_call_cudnn = True
        if self.group > 1 and cuda.cuda.cudnn.getVersion() < 7000:
            self.should_call_cudnn = False

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
                mlibcudnn_conv.convolutionBackwardFilter_v3.call_count,
                self.should_call_cudnn)
            self.assertFalse(
                mlibcudnn_deconv.getConvolutionBackwardDataAlgorithm.called)
            self.assertEqual(
                mlibcudnn_deconv.convolutionBackwardData_v3.call_count,
                self.should_call_cudnn)

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
                x_data, W_data, b_data, gy_data = tuple(
                    cuda.to_gpu(data) for data in self._contiguous(
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
                             cover_all=False, group=self.group)
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
