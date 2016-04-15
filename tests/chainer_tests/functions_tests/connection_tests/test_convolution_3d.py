import unittest

import mock
import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer.functions.connection import convolution_3d
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import conv

def _triplet(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x, x


def _asfortranarray(x):
    xp = cuda.get_array_module(x)
    if xp is numpy:
        return xp.asfortranarray(x)
    else:
        return xp.ascontiguousarray(x.T).T




@testing.parameterize(
    *testing.product({
        'in_channels': [3],
        'out_channels': [2],
        'wscale': [1],
        'ksize': [3],
        'stride': [1, 2],
        'pad': [1],
        'nobias': [True, False],
        'use_cudnn': [True],
        'c_contiguous': [True, False],
    })
)
class TestConvolution3DFunction(unittest.TestCase):

    def setUp(self, use_cudnn=True):
        kh, kw, kd = _triplet(self.ksize)
        sh, sw, sd = _triplet(self.stride)
        ph, pw, pd = _triplet(self.pad)
        self.use_cudnn = use_cudnn
        self.W = numpy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * kd * self.in_channels)),
            (self.out_channels, self.in_channels, kh, kw, kd)).astype(numpy.float32)
        self.b = numpy.random.uniform(
            -1, 1, self.out_channels).astype(numpy.float32)

        in_h, in_w, in_d = (4, 4, 4)
        out_h = conv.get_conv_outsize(in_h, kh, sh, ph)
        out_w = conv.get_conv_outsize(in_w, kw, sw, pw)
        out_d = conv.get_conv_outsize(in_d, kd, sd, pd)
        self.x = numpy.random.uniform(-1, 1,
                                      (2, self.in_channels, in_h, in_w, in_d)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1,
                                       (2, self.out_channels, out_h, out_w, out_d)).astype(numpy.float32)


    @attr.cudnn
    def test_forward_consistency(self, nobias=False):
        x_cpu = chainer.Variable(self.x)
        W_cpu = chainer.Variable(self.W)
        b_cpu = None if nobias else chainer.Variable(self.b)
        y_cpu = functions.convolution_3d(
            x_cpu, W_cpu, b_cpu, stride=self.stride, pad=self.pad,
            use_cudnn=self.use_cudnn)

        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        W_gpu = chainer.Variable(cuda.to_gpu(self.W))
        b_gpu = None if nobias else chainer.Variable(cuda.to_gpu(self.b))
        y_gpu = functions.convolution_3d(
            x_gpu, W_gpu, b_gpu, stride=self.stride, pad=self.pad,
            use_cudnn=self.use_cudnn)

        gradient_check.assert_allclose(y_cpu.data, y_gpu.data.get())




    def check_backward(self, x_data, W_data, b_data, y_grad):
        if not self.c_contiguous:
            x_data = _asfortranarray(x_data)
            W_data = _asfortranarray(W_data)
            y_grad = _asfortranarray(y_grad)
            self.assertFalse(x_data.flags.c_contiguous)
            self.assertFalse(W_data.flags.c_contiguous)
            self.assertFalse(y_grad.flags.c_contiguous)
            if b_data is not None:
                xp = cuda.get_array_module(b_data)
                b = xp.empty((len(b_data) * 2,), dtype=self.b.dtype)
                b[::2] = b_data
                b_data = b[::2]
                self.assertFalse(b_data.flags.c_contiguous)
        args = (x_data, W_data)
        if b_data is not None:
            args = args + (b_data,)

        gradient_check.check_backward(
            convolution_3d.Convolution3DFunction(
                self.stride, self.pad, self.use_cudnn),
            args, y_grad, eps=1e-3, atol=1e-3, rtol=1e-3)


    @condition.retry(3)
    def test_backward_cpu(self):
        b = None if self.nobias else self.b
        self.check_backward(self.x, self.W, self.b, self.gy)


    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        b = None if self.nobias else cuda.to_gpu(self.b)
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            b, cuda.to_gpu(self.gy))





testing.run_module(__name__, __file__)
