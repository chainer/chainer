import unittest

import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import array
from chainer.testing import attr
from chainer.testing import backend
from chainer.testing import condition
from chainer.testing import parameterize
from chainer.utils import conv


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


@parameterize(*(testing.product([
    testing.product({
        'c_contiguous': [True],
        'test_outsize': [True, False],
        'nobias': [True],
        'stride': [1, 2],
        'dilate': [1],
        'x_dtype': [numpy.float32],
        'W_dtype': [numpy.float32],
        'groups': [1, 2],
    })
    + testing.product({
        'c_contiguous': [False],
        'test_outsize': [True],
        'nobias': [False],
        'stride': [1, 2],
        'dilate': [1, 2],
        'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
        'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
        'groups': [1, 2],
    }),
])))
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
        })])
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    }))
class TestDeconvolution2DFunction(unittest.TestCase):

    def setUp(self):
        in_channels_a_group = 3
        out_channels_a_group = 2
        self.in_channels = in_channels_a_group * self.groups
        self.out_channels = out_channels_a_group * self.groups
        self.ksize = 3
        self.pad = 1
        kh, kw = _pair(self.ksize)
        sh, sw = _pair(self.stride)
        ph, pw = _pair(self.pad)

        W = numpy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * in_channels_a_group)),
            (self.in_channels, out_channels_a_group, kh, kw)
        ).astype(self.W_dtype)
        b = None if self.nobias else numpy.random.uniform(
            -1, 1, self.out_channels).astype(self.x_dtype)

        N = 2
        inh, inw = 4, 3
        outh = conv.get_deconv_outsize(inh, kh, sh, ph, d=self.dilate)
        outw = conv.get_deconv_outsize(inw, kw, sw, pw, d=self.dilate)
        self.outsize = (outh, outw) if self.test_outsize else None
        x = numpy.random.uniform(
            -1, 1, (N, self.in_channels, inh, inw)).astype(self.x_dtype)
        gy = numpy.random.uniform(
            -1, 1, (N, self.out_channels, outh, outw)).astype(self.x_dtype)

        ggx = numpy.random.uniform(-1, 1, x.shape).astype(
            self.x_dtype)
        ggW = numpy.random.uniform(-1, 1, W.shape).astype(
            self.W_dtype)
        ggb = None if self.nobias else numpy.random.uniform(
            -1, 1, b.shape).astype(self.x_dtype)

        self.inputs = [x, W, b]
        self.grad_outputs = [gy]
        self.grad_grad_inputs = [ggx, ggW, ggb]

        self.test_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        self.check_double_backward_options = {'dtype': numpy.float64}
        if self.x_dtype == numpy.float16:
            self.test_forward_options.update(atol=5e-3, rtol=5e-2)
            self.check_backward_options.update(atol=5e-4, rtol=5e-3)
            self.check_double_backward_options.update(atol=5e-3, rtol=5e-2)
        elif self.W_dtype == numpy.float16:
            self.check_backward_options.update(atol=5e-4, rtol=5e-3)
            self.check_double_backward_options.update(atol=5e-3, rtol=5e-2)

    def forward_cpu(self, inputs):
        x, W, b = inputs
        x_cpu = chainer.Variable(x)
        W_cpu = chainer.Variable(W)
        b_cpu = None if b is None else chainer.Variable(b)
        with chainer.using_config('use_ideep', 'never'):
            y_cpu = F.deconvolution_2d(
                x_cpu, W_cpu, b_cpu, stride=self.stride, pad=self.pad,
                outsize=self.outsize, dilate=self.dilate, groups=self.groups)
        return y_cpu,

    def check_forward(self, inputs, backend_config):
        y_expected, = self.forward_cpu(inputs)

        x, W, b = backend_config.get_array(inputs)
        x = chainer.Variable(x)
        W = chainer.Variable(W)
        b = None if b is None else chainer.Variable(b)

        with backend_config:
            y_actual = F.deconvolution_2d(
                x, W, b, stride=self.stride, pad=self.pad,
                outsize=self.outsize, dilate=self.dilate, groups=self.groups)

        assert y_expected.data.dtype == self.x_dtype
        assert y_actual.data.dtype == self.x_dtype
        testing.assert_allclose(
            y_expected.data, y_actual.data, **self.test_forward_options)

    @attr.gpu
    def test_forward(self, backend_config):
        self.check_forward(self.inputs, backend_config)

    def check_backward(self, inputs, grad_outputs, backend_config):
        inputs = backend_config.get_array(inputs)
        grad_outputs = backend_config.get_array(grad_outputs)

        if not self.c_contiguous:
            inputs = array._as_noncontiguous_array(inputs)
            grad_outputs = array._as_noncontiguous_array(grad_outputs)

        x_data, W_data, b_data = inputs
        y_grad, = grad_outputs

        args = (x_data, W_data)
        if b_data is not None:
            args = args + (b_data,)

        def f(*args):
            return F.deconvolution_2d(
                *args, stride=self.stride, pad=self.pad, outsize=self.outsize,
                dilate=self.dilate, groups=self.groups)

        with backend_config:
            gradient_check.check_backward(
                f, args, y_grad, **self.check_backward_options)

    @condition.retry(10)
    def test_backward(self, backend_config):
        self.check_backward(self.inputs, self.grad_outputs, backend_config)

    def check_double_backward(
            self, inputs, grad_outputs, grad_grad_inputs, backend_config):
        inputs = backend_config.get_array(inputs)
        grad_outputs = backend_config.get_array(grad_outputs)
        grad_grad_inputs = backend_config.get_array(grad_grad_inputs)

        if not self.c_contiguous:
            inputs = array._as_noncontiguous_array(inputs)
            grad_outputs = array._as_noncontiguous_array(grad_outputs)
            grad_grad_inputs = array._as_noncontiguous_array(grad_grad_inputs)

        x_data, W_data, b_data = inputs
        y_grad, = grad_outputs
        x_grad_grad, W_grad_grad, b_grad_grad = grad_grad_inputs

        args = (x_data, W_data)
        grad_grads = (x_grad_grad, W_grad_grad)
        if b_data is not None:
            args = args + (b_data,)
            grad_grads = grad_grads + (b_grad_grad,)

        def f(*args):
            return F.deconvolution_2d(
                *args, stride=self.stride, pad=self.pad, outsize=self.outsize,
                dilate=self.dilate, groups=self.groups)

        with backend_config:
            gradient_check.check_double_backward(
                f, args, y_grad, grad_grads,
                **self.check_double_backward_options)

    @condition.retry(10)
    def test_double_backward(self, backend_config):
        self.check_double_backward(self.inputs, self.grad_outputs,
                                   self.grad_grad_inputs, backend_config)


@testing.parameterize(*testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
    'cudnn_deterministic': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'groups': [1, 2],
}))
@attr.cudnn
class TestDeconvolution2DCudnnCall(unittest.TestCase):

    def setUp(self):
        in_channels_a_group = 3
        out_channels_a_group = 2
        self.in_channels = in_channels_a_group * self.groups
        self.out_channels = out_channels_a_group * self.groups
        kh, kw = _pair(3)
        sh, sw = _pair(1)
        ph, pw = _pair(1)
        self.W = cuda.cupy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * in_channels_a_group)),
            (self.in_channels, out_channels_a_group, kh, kw)
        ).astype(self.dtype)
        N = 2
        inh, inw = 4, 3
        outh = conv.get_deconv_outsize(inh, kh, sh, ph)
        outw = conv.get_deconv_outsize(inw, kw, sw, pw)
        self.x = cuda.cupy.random.uniform(
            -1, 1, (N, self.in_channels, inh, inw)).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(
            -1, 1, (N, self.out_channels, outh, outw)).astype(self.dtype)
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.should_call_cudnn = chainer.should_use_cudnn('>=auto')
            if self.groups > 1 and cuda.cuda.cudnn.getVersion() < 7000:
                self.should_call_cudnn = False

    def forward(self):
        x = chainer.Variable(self.x)
        W = chainer.Variable(self.W)
        return F.deconvolution_2d(x, W, None, stride=1, pad=1,
                                  groups=self.groups)

    def test_call_cudnn_forward(self):
        name = 'cupy.cudnn.convolution_backward_data'
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with chainer.using_config('cudnn_deterministic',
                                      self.cudnn_deterministic):
                with testing.patch(name) as func:
                    self.forward()
                self.assertEqual(func.called, self.should_call_cudnn)

    def test_call_cudnn_backward(self):
        data_func_name = 'cupy.cudnn.convolution_forward'
        filter_func_name = 'cupy.cudnn.convolution_backward_filter'

        with chainer.using_config('use_cudnn', self.use_cudnn):
            with chainer.using_config('cudnn_deterministic',
                                      self.cudnn_deterministic):
                y = self.forward()
                y.grad = self.gy
                with testing.patch(data_func_name) as data_func, \
                        testing.patch(filter_func_name) as filter_func:
                    y.backward()
                    self.assertEqual(
                        data_func.called, self.should_call_cudnn)
                    self.assertEqual(
                        filter_func.called, self.should_call_cudnn)


@testing.parameterize(*testing.product({
    'c_contiguous': [True, False],
    'cudnn_deterministic': [True, False],
    'nobias': [True, False],
    'groups': [1, 2],
}))
@attr.gpu
@attr.cudnn
class TestDeconvolution2DFunctionCudnnDeterministic(unittest.TestCase):

    def setUp(self):
        self.stride = 2
        self.pad = 1
        batch_sz = 2
        in_channels_a_group = 64
        out_channels_a_group = 64
        in_channels = in_channels_a_group * self.groups
        out_channels = out_channels_a_group * self.groups
        kh, kw = (3, 3)
        in_h, in_w = (32, 128)
        out_h, out_w = (63, 255)
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

    def test_cudnn_deterministic(self):
        x1, W1, b1, y1 = self._run()
        x2, W2, b2, y2 = self._run()

        cuda.cupy.testing.assert_array_equal(x1.grad, x2.grad)
        cuda.cupy.testing.assert_array_equal(y1.data, y2.data)
        cuda.cupy.testing.assert_array_equal(W1.grad, W2.grad)

    def _contiguous(self, *inputs):
        if self.c_contiguous:
            return inputs
        else:
            return array._as_noncontiguous_array(inputs)

    def _run(self):
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
        with chainer.using_config('use_cudnn', 'always'):
            y = F.deconvolution_2d(x, W, b, stride=self.stride, pad=self.pad,
                                   groups=self.groups)
        return x, W, b, y


testing.run_module(__name__, __file__)
