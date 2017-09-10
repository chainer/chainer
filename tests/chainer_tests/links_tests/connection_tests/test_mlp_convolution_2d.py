import unittest

import mock
import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import links
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'use_cudnn': 'always'},
    {'use_cudnn': 'never'},
)
class TestMLPConvolution2D(unittest.TestCase):

    def setUp(self):
        self.mlp = links.MLPConvolution2D(
            3, (96, 96, 96), 11, activation=functions.sigmoid)
        self.x = numpy.zeros((10, 3, 20, 20), dtype=numpy.float32)

    def test_init(self):
        self.assertIs(self.mlp.activation, functions.sigmoid)

        self.assertEqual(len(self.mlp), 3)
        for i, conv in enumerate(self.mlp):
            self.assertIsInstance(conv, links.Convolution2D)
            if i == 0:
                self.assertEqual(conv.W.data.shape, (96, 3, 11, 11))
            else:
                self.assertEqual(conv.W.data.shape, (96, 96, 1, 1))

    def check_call(self, x_data):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            x = chainer.Variable(x_data)
            actual = self.mlp(x)
            act = functions.sigmoid
            expect = self.mlp[2](act(self.mlp[1](act(self.mlp[0](x)))))
        numpy.testing.assert_array_equal(
            cuda.to_cpu(expect.data), cuda.to_cpu(actual.data))

    def test_call_cpu(self):
        self.check_call(self.x)

    @attr.gpu
    def test_call_gpu(self):
        self.mlp.to_gpu()
        self.check_call(cuda.to_gpu(self.x))


@testing.parameterize(
    {'use_cudnn': 'always'},
    {'use_cudnn': 'never'},
)
@attr.cudnn
class TestMLPConvolution2DCudnnCall(unittest.TestCase):

    def setUp(self):
        self.mlp = links.MLPConvolution2D(
            3, (96, 96, 96), 11, activation=functions.sigmoid)
        self.mlp.to_gpu()
        self.x = cuda.cupy.zeros((10, 3, 20, 20), dtype=numpy.float32)
        self.gy = cuda.cupy.zeros((10, 96, 10, 10), dtype=numpy.float32)

    def forward(self):
        x = chainer.Variable(self.x)
        return self.mlp(x)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with mock.patch('cupy.cuda.cudnn.convolutionForward') as func:
                self.forward()
                self.assertEqual(func.called,
                                 chainer.should_use_cudnn('>=auto'))

    def test_call_cudnn_backrward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            y.grad = self.gy
            patch = 'cupy.cuda.cudnn.convolutionBackwardData_v3'
            with mock.patch(patch) as func:
                y.backward()
                self.assertEqual(func.called,
                                 chainer.should_use_cudnn('>=auto'))


@testing.parameterize(*testing.product({
    'use_cudnn': ['always', 'never'],
    'mlpconv_args': [
        ((None, (96, 96, 96), 11), {'activation': functions.sigmoid}),
        (((96, 96, 96), 11), {'activation': functions.sigmoid})
    ]
}))
class TestMLPConvolution2DShapePlaceholder(unittest.TestCase):

    def setUp(self):
        args, kwargs = self.mlpconv_args
        self.mlp = links.MLPConvolution2D(*args, **kwargs)
        self.x = numpy.zeros((10, 3, 20, 20), dtype=numpy.float32)

    def test_init(self):
        self.assertIs(self.mlp.activation, functions.sigmoid)
        self.assertEqual(len(self.mlp), 3)

    def check_call(self, x_data):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            x = chainer.Variable(x_data)
            actual = self.mlp(x)
            act = functions.sigmoid
            expect = self.mlp[2](act(self.mlp[1](act(self.mlp[0](x)))))
        numpy.testing.assert_array_equal(
            cuda.to_cpu(expect.data), cuda.to_cpu(actual.data))
        for i, conv in enumerate(self.mlp):
            self.assertIsInstance(conv, links.Convolution2D)
            if i == 0:
                self.assertEqual(conv.W.data.shape, (96, 3, 11, 11))
            else:
                self.assertEqual(conv.W.data.shape, (96, 96, 1, 1))

    def test_call_cpu(self):
        self.check_call(self.x)

    @attr.gpu
    def test_call_gpu(self):
        self.mlp.to_gpu()
        self.check_call(cuda.to_gpu(self.x))


class TestInitArgumentForv2(unittest.TestCase):

    in_channels = 10
    out_channels = (15, 20)
    ksize = 3
    stride = 1
    pad = 0

    def test_valid_instantiation_ksize_is_not_none(self):
        l = links.MLPConvolution2D(
            self.in_channels, self.out_channels, self.ksize, self.stride,
            self.pad, functions.relu, conv_init=None, bias_init=None)
        self.assertEqual(len(l), 2)
        self.assertEqual(l[0].W.shape,
                         (self.out_channels[0], self.in_channels,
                          self.ksize, self.ksize))
        self.assertEqual(l[1].W.shape,
                         (self.out_channels[1], self.out_channels[0], 1, 1))

    def test_valid_instantiation_ksize_is_none(self):
        l = links.MLPConvolution2D(self.out_channels, self.ksize, None,
                                   self.stride, self.pad, functions.relu,
                                   conv_init=None, bias_init=None)
        x = numpy.random.uniform(
            -1, 1, (10, self.in_channels, 10, 10)).astype(numpy.float32)
        l(x)  # create weight tensors of convolutions by initialization

        self.assertEqual(len(l), 2)
        self.assertEqual(l[0].W.shape,
                         (self.out_channels[0], self.in_channels,
                          self.ksize, self.ksize))
        self.assertEqual(l[1].W.shape,
                         (self.out_channels[1], self.out_channels[0], 1, 1))

    def test_valid_instantiation_in_channels_is_omitted(self):
        l = links.MLPConvolution2D(
            self.out_channels, self.ksize, stride=self.stride, pad=self.pad,
            activation=functions.relu, conv_init=None, bias_init=None)
        x = numpy.random.uniform(
            -1, 1, (10, self.in_channels, 10, 10)).astype(numpy.float32)
        l(x)  # create weight tensors of convolutions by initialization

        self.assertEqual(len(l), 2)
        self.assertEqual(l[0].W.shape,
                         (self.out_channels[0], self.in_channels,
                          self.ksize, self.ksize))
        self.assertEqual(l[1].W.shape,
                         (self.out_channels[1], self.out_channels[0], 1, 1))

    def test_forbid_wscale_as_a_positional_argument(self):
        with self.assertRaises(TypeError):
            # 7th positional argument was wscale in v1
            links.MLPConvolution2D(self.in_channels, self.out_channels, None,
                                   self.stride, self.pad, functions.relu, 1)

    def test_forbid_wscale_as_a_keyword_argument(self):
        with self.assertRaises(ValueError):
            links.MLPConvolution2D(
                self.in_channels, self.out_channels, wscale=1)


testing.run_module(__name__, __file__)
