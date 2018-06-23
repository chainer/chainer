import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import conv

import chainer.functions as F
import numpy
import unittest


@testing.parameterize(*testing.product({
    'in_shape': [(4, 3, 6, 8), (4, 3, 5, 7)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestUpsampling2D(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.in_shape).astype(self.dtype)
        self.ksize = 2
        self.stride = 2
        with chainer.using_config('use_cudnn', 'never'):
            self.pooled_y, self.indices = F.max_pooling_2d(
                self.x, ksize=self.ksize, stride=self.stride,
                return_indices=True)
        self.gy = numpy.random.uniform(
            -1, 1, self.in_shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(
            -1, 1, self.pooled_y.shape).astype(self.dtype)
        self.check_backward_options = {}
        self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        if self.dtype == numpy.float16:
            self.check_double_backward_options = {'atol': 3e-3, 'rtol': 3e-2}

    def check_forward(self, y):
        y = F.upsampling_2d(
            self.pooled_y, self.indices, ksize=self.ksize,
            stride=self.stride, outsize=self.in_shape[2:])
        if isinstance(y.array, numpy.ndarray):
            y = conv.im2col_cpu(
                y.array, self.ksize, self.ksize, self.stride, self.stride,
                0, 0)
        else:
            y = conv.im2col_gpu(
                y.array, self.ksize, self.ksize, self.stride, self.stride,
                0, 0)
        for i in numpy.ndindex(y.shape):
            n, c, ky, kx, oy, ox = i
            up_y = y[n, c, ky, kx, oy, ox]
            if ky * y.shape[3] + kx == self.indices[n, c, oy, ox]:
                in_y = self.pooled_y.array[n, c, oy, ox]
                testing.assert_allclose(in_y, up_y)
            else:
                testing.assert_allclose(up_y, 0)

    def test_forward_cpu(self):
        self.pooled_y.to_cpu()
        self.check_forward(self.pooled_y)

    @attr.gpu
    def test_forward_gpu(self):
        self.pooled_y.to_gpu()
        self.check_forward(self.pooled_y)

    def check_backward(self, x_data, y_grad):
        def f(x):
            return F.upsampling_2d(
                x, self.indices, ksize=self.ksize, stride=self.stride,
                outsize=self.in_shape[2:])
        gradient_check.check_backward(
            f, x_data, y_grad, dtype='d', **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.pooled_y.array, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(
            self.pooled_y.array), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad,
                              use_cudnn='always'):
        def f(x):
            y = F.upsampling_2d(
                x, self.indices, ksize=self.ksize,
                stride=self.stride, outsize=self.in_shape[2:])
            return y * y
        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_double_backward(
                f, x_data, y_grad, x_grad_grad, dtype='d',
                **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.pooled_y.array, self.gy, self.ggx, 'never')

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.pooled_y.array), cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx))

    @attr.gpu
    def test_double_backward_gpu_non_contiguous(self):
        self.check_double_backward(
            cuda.cupy.asfortranarray(cuda.to_gpu(self.pooled_y.array)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.ggx)))

    @attr.gpu
    def test_double_backward_gpu_no_cudnn(self):
        self.check_double_backward(
            cuda.to_gpu(self.pooled_y.array), cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx), 'never')


testing.run_module(__name__, __file__)
