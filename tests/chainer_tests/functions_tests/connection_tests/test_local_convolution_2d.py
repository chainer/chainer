from six import moves
import unittest

import numpy

from chainer import cuda
from chainer.functions.connection import convolution_2d
from chainer.functions.connection import local_convolution_2d
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*(testing.product({
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
class TestLocalConvolution2DFunction(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 6
        kh, kw = (3, 3)
        oh, ow = (2, 2)
        self.stride = 1
        self.W = numpy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * in_channels)),
            (out_channels, oh, ow, in_channels, kh, kw)).astype(self.W_dtype)
        self.b = numpy.random.uniform(
            -1, 1, (out_channels, oh, ow,)).astype(self.x_dtype)

        self.x = numpy.random.uniform(
            -1, 1, (2, in_channels, 4, 4)).astype(self.x_dtype)
        self.gy = numpy.random.uniform(
            -1, 1, (2, out_channels, oh, ow)).astype(self.x_dtype)
        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.x_dtype == numpy.float16 or self.W_dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-3, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-3, 'rtol': 5e-3}

    def check_forward(self, x_data, W_data, b_data):
        # If all the filters are the same,
        # the operation is equivalent to convolution_2d
        for i in moves.range(W_data.shape[1]):
            for j in moves.range(W_data.shape[2]):
                W_data[:, i, j, ...] = W_data[:, 0, 0, ...]
        args1 = (x_data, W_data)
        args2 = (x_data, W_data[:, 0, 0, ...])
        if b_data is not None:
            for i in moves.range(b_data.shape[1]):
                for j in moves.range(b_data.shape[2]):
                    b_data[:, i, j, ] = b_data[:, 0, 0, ]
            args1 = args1 + (b_data,)
            b_data2 = b_data[:, 0, 0, ]
            args2 = args2 + (b_data2,)

        f1 = local_convolution_2d.LocalConvolution2DFunction(self.stride)
        y1 = f1.apply(args1)[0].data

        f2 = convolution_2d.Convolution2DFunction(self.stride, 0)
        y2 = f2.apply(args2)[0].data
        testing.assert_allclose(y1, y2, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.W, self.b)

    def test_forward_cpu_nobias(self):
        self.check_forward(self.x, self.W, None)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                           cuda.to_gpu(self.b))

    @attr.gpu
    def test_forward_gpu_nobias(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.W), None)

    def check_backward(self, x_data, W_data, b_data, y_grad):
        args = (x_data, W_data)
        if b_data is not None:
            args = args + (b_data,)

        gradient_check.check_backward(
            local_convolution_2d.local_convolution_2d,
            args, y_grad, **self.check_backward_options)

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


testing.run_module(__name__, __file__)
