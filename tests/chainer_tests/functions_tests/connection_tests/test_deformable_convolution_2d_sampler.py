import unittest

import numpy

import chainer
from chainer import cuda
from chainer.functions import convolution_2d
from chainer.functions import deformable_convolution_2d_sampler
from chainer import utils

from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'params': [
        (1, 1, 1, 1, 1, 1),
        (2, 2, 2, 2, 2, 2),
        (1, 2, 2, 1, 1, 2),
        (1, 2, 3, 4, 1, 2),
        (1, 2, 3, 4, 4, 5),
        (3, 3, 2, 2, 1, 1),
    ],
    'use_cudnn': ['always', 'never']
}))
class TestDeformableConvolution2DSamplerFunctionZeroOffset(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 2
        batch_size = 2
        h = 9
        w = 9

        kh, kw, sy, sx, ph, pw = self.params

        self.stride = (sy, sx)
        self.pad = (ph, pw)

        self.W = numpy.random.normal(
            size=(out_channels, in_channels, kh, kw)).astype(numpy.float32)
        self.b = numpy.random.uniform(
            size=(out_channels,)).astype(numpy.float32)

        self.x = numpy.random.uniform(
            size=(batch_size, in_channels, h, w)).astype(numpy.float32)

        out_h = utils.conv.get_conv_outsize(h, kh, sy, ph)
        out_w = utils.conv.get_conv_outsize(w, kw, sx, pw)
        self.offset = numpy.zeros(
            (batch_size, 2 * kh * kw, out_h, out_w), dtype=numpy.float32)

    def check_forward(self, x, offset, W, b, stride, pad):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            x = chainer.Variable(x)
            offset = chainer.Variable(offset)
            out = deformable_convolution_2d_sampler(
                x, offset, W, b, stride, pad).data
            expeceted = convolution_2d(
                x, W, b, stride, pad).data
        testing.assert_allclose(out, expeceted)

    def test_forward_cpu(self):
        self.check_forward(
            self.x, self.offset, self.W, self.b, self.stride, self.pad)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x),
            cuda.to_gpu(self.offset),
            cuda.to_gpu(self.W),
            cuda.to_gpu(self.b),
            self.stride, self.pad)


@testing.parameterize(*testing.product({
    'params': [
        (1, 1, 1, 1, 1, 1),
        (2, 2, 2, 2, 2, 2),
        (1, 2, 2, 1, 1, 2),
        (1, 2, 3, 4, 1, 2),
        (1, 2, 3, 4, 4, 5),
        (3, 3, 2, 2, 1, 1),
    ],
    'use_cudnn': ['always', 'never']
}))
class TestDeformableConvolution2DSamplerFunctionLeftBottomOffset(
        unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 2
        batch_size = 2
        h = 9
        w = 9

        kh, kw, sy, sx, ph, pw = self.params

        self.stride = (sy, sx)
        self.pad = (ph, pw)

        self.W = numpy.random.normal(
            size=(out_channels, in_channels, kh, kw)).astype(numpy.float32)
        self.b = numpy.random.uniform(
            size=(out_channels,)).astype(numpy.float32)

        self.x = numpy.random.uniform(
            size=(batch_size, in_channels, h, w)).astype(numpy.float32)

        out_h = utils.conv.get_conv_outsize(h, kh, sy, ph)
        out_w = utils.conv.get_conv_outsize(w, kw, sx, pw)
        self.offset = numpy.zeros(
            (batch_size, 2 * kh * kw, out_h, out_w), dtype=numpy.float32)

    def check_forward(self, x, offset, W, b, stride, pad):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            _, _, h, w = x.shape
            _, _, kh, kw = W.shape
            offset[:, :kh * kw] = -1 * stride[1]
            offset[:, kh * kw:] = 1 * stride[0]

            x = chainer.Variable(x)
            offset = chainer.Variable(offset)
            out = deformable_convolution_2d_sampler(
                x, offset, W, b, stride, pad).data
            pad = (pad[0] + 1 * stride[0], pad[1] + 1 * stride[1])
            expeceted = convolution_2d(
                x, W, b, stride, pad).data
            expeceted = expeceted[:, :, 2:, :-2]
        testing.assert_allclose(out, expeceted)

    def test_forward_cpu(self):
        self.check_forward(
            self.x, self.offset, self.W, self.b, self.stride, self.pad)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x),
            cuda.to_gpu(self.offset),
            cuda.to_gpu(self.W),
            cuda.to_gpu(self.b),
            self.stride, self.pad)


testing.run_module(__name__, __file__)
