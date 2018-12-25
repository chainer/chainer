import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'ratio': [0.0, 0.9],
    'train': [True, False],
    'use_batchwise_mask': [True, False],
}))
class TestSimplifiedDropconnect(unittest.TestCase):

    def setUp(self):
        self.W = numpy.random.uniform(
            -1, 1, (2, 3)).astype(self.W_dtype)
        self.b = numpy.random.uniform(
            -1, 1, 2).astype(self.x_dtype)

        self.x = numpy.random.uniform(-1, 1, (4, 3)).astype(self.x_dtype)
        self.gy = numpy.random.uniform(-1, 1, (4, 2)).astype(self.x_dtype)
        self.ggW = numpy.random.uniform(
            -1, 1, (2, 3)).astype(self.W_dtype)
        self.ggb = numpy.random.uniform(
            -1, 1, 2).astype(self.x_dtype)
        self.ggx = numpy.random.uniform(-1, 1, (4, 3)).astype(self.x_dtype)
        self.y = self.x.dot(self.W.T) + self.b
        self.check_forward_options = {}
        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-3}
        self.check_double_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
        if self.x_dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-2, 'rtol': 5e-2}
        elif self.W_dtype == numpy.float16:
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-1, 'rtol': 1e-1}

    def check_forward(self, x_data, W_data, b_data):
        # Check only data type, y is tested by SimplifiedDropconnect link test.
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        if b_data is None:
            y = functions.simplified_dropconnect(x, W, None,
                                                 self.ratio, self.train, None,
                                                 self.use_batchwise_mask)
        else:
            b = chainer.Variable(b_data)
            y = functions.simplified_dropconnect(x, W, b,
                                                 self.ratio, self.train, None,
                                                 self.use_batchwise_mask)
        self.assertEqual(y.data.dtype, self.x_dtype)
        mask = y.creator.mask
        mask = cuda.to_cpu(mask)
        if self.use_batchwise_mask:
            self.assertEqual(mask.shape, (x.shape[0],) + W.shape)
        else:
            self.assertEqual(mask.shape, W.shape)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.W, self.b)

    def test_forward_cpu_nobias(self):
        self.check_forward(self.x, self.W, None)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W), cuda.to_gpu(self.b))

    @attr.gpu
    def test_forward_gpu_nobias(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W), None)

    def check_backward(self, x_data, W_data, b_data, y_grad):
        args = x_data, W_data
        if b_data is not None:
            args += b_data,

        if self.use_batchwise_mask:
            mask_shape = (x_data.shape[0],) + W_data.shape
        else:
            mask_shape = W_data.shape

        xp = backend.get_array_module(x_data)
        mask = xp.random.rand(*mask_shape) >= self.ratio

        def f(x, W, b=None):
            return functions.simplified_dropconnect(
                x, W, b, self.ratio, self.train, mask,
                self.use_batchwise_mask)

        gradient_check.check_backward(
            f, args, y_grad, eps=1e-2, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    def test_backward_cpu_nobias(self):
        self.check_backward(self.x, self.W, None, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy))

    @attr.gpu
    def test_backward_gpu_nobias(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy))

    def check_double_backward(
            self, x_data, W_data, b_data, y_grad,
            x_grad_grad, W_grad_grad, b_grad_grad):
        args = x_data, W_data
        grads = x_grad_grad, W_grad_grad
        if b_data is not None:
            args += b_data,
            grads += b_grad_grad,

        if self.use_batchwise_mask:
            mask_shape = (x_data.shape[0],) + W_data.shape
        else:
            mask_shape = W_data.shape

        xp = backend.get_array_module(x_data)
        mask = xp.random.rand(*mask_shape) >= self.ratio

        def f(x, W, b=None):
            return functions.simplified_dropconnect(
                x, W, b, self.ratio, self.train, mask,
                self.use_batchwise_mask)

        gradient_check.check_double_backward(
            f, args, y_grad, grads, eps=1e-2,
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.x, self.W, self.b, self.gy, self.ggx, self.ggW, self.ggb)

    def test_double_backward_cpu_nobias(self):
        self.check_double_backward(
            self.x, self.W, None, self.gy, self.ggx, self.ggW, None)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W), cuda.to_gpu(self.b),
            cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx),
            cuda.to_gpu(self.ggW), cuda.to_gpu(self.ggb))

    @attr.gpu
    def test_double_backward_gpu_nobias(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W), None,
            cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx),
            cuda.to_gpu(self.ggW), None)


testing.run_module(__name__, __file__)
