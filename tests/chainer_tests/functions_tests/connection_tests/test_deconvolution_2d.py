import unittest

import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import testing
from chainer.testing import array
from chainer.testing import attr
from chainer.testing import backend
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
    None,
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
class TestDeconvolution2DFunction(testing.FunctionTestCase):

    def setUp(self):
        self.in_channels_a_group = 3
        self.out_channels_a_group = 2
        self.in_channels = self.in_channels_a_group * self.groups
        self.out_channels = self.out_channels_a_group * self.groups
        self.ksize = 3
        self.pad = 1
        self.kh, self.kw = _pair(self.ksize)
        self.sh, self.sw = _pair(self.stride)
        self.ph, self.pw = _pair(self.pad)

        self.N = 2
        self.inh, self.inw = 4, 3
        self.outh = conv.get_deconv_outsize(self.inh, self.kh, self.sh,
                                            self.ph, d=self.dilate)
        self.outw = conv.get_deconv_outsize(self.inw, self.kw, self.sw,
                                            self.pw, d=self.dilate)
        self.outsize = (self.outh, self.outw) if self.test_outsize else None

        self.check_backward_options.update(atol=5e-4, rtol=5e-3)
        if self.x_dtype == numpy.float16:
            self.check_forward_options.update(atol=5e-3, rtol=5e-2)
            self.check_double_backward_options.update(atol=5e-3, rtol=5e-2)
        elif self.W_dtype == numpy.float16:
            self.check_double_backward_options.update(atol=5e-3, rtol=5e-2)

    def generate_inputs(self):
        x = numpy.random.uniform(
            -1, 1, (self.N, self.in_channels, self.inh, self.inw)
        ).astype(self.x_dtype)
        W = numpy.random.normal(
            0, numpy.sqrt(1. / (self.kh * self.kw * self.in_channels_a_group)),
            (self.in_channels, self.out_channels_a_group, self.kh, self.kw)
        ).astype(self.W_dtype)
        b = None if self.nobias else numpy.random.uniform(
            -1, 1, self.out_channels).astype(self.x_dtype)
        if self.nobias:
            return x, W
        return x, W, b

    def forward_expected(self, inputs):
        if self.nobias:
            x, W = inputs
            b = None
        else:
            x, W, b = inputs
        with chainer.using_config('use_ideep', 'never'):
            y_cpu = F.deconvolution_2d(
                x, W, b, stride=self.stride, pad=self.pad,
                outsize=self.outsize, dilate=self.dilate, groups=self.groups)
        return y_cpu.array,

    def forward(self, inputs, device):
        if self.nobias:
            x, W = inputs
            b = None
        else:
            x, W, b = inputs
        y = F.deconvolution_2d(
            x, W, b, stride=self.stride, pad=self.pad,
            outsize=self.outsize, dilate=self.dilate, groups=self.groups)
        return y,


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
