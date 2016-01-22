import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


class TestSpatialPyramidPooling2D(unittest.TestCase):
    pyramid_height = 3
    output_dim = 63  # channels(c=3) * (1 + 4 + 16) = 63
    n, c, h, w = 2, 3, 9, 8
    pooling_class = functions.MaxPooling2D

    def setUp(self):
        # Spacial pyramid pooling uses max pooling in its implementation.
        # To avoid unstability of numerical gradient, use enoufh different
        # values.
        shape = (self.n, self.c, self.h, self.w)
        size = numpy.prod(shape)
        self.x = numpy.arange(size, dtype=numpy.float32).reshape(shape)
        numpy.random.shuffle(self.x)
        self.x += numpy.random.uniform(
            0.4, 0.6, shape).astype(numpy.float32)
        self.x /= size

        self.one = numpy.ones(
            (self.n, self.c, self.h, self.w)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (self.n, self.output_dim, 1, 1))
        self.gy = self.gy.astype(numpy.float32)

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.spatial_pyramid_pooling_2d(
            x, self.pyramid_height, self.pooling_class,
            use_cudnn=use_cudnn)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)

    def check_forward_ones(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.spatial_pyramid_pooling_2d(
            x, self.pyramid_height, self.pooling_class, use_cudnn=use_cudnn)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual((self.n, self.output_dim, 1, 1), y_data.shape)
        gradient_check.assert_allclose(y_data, numpy.ones_like(y_data))

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)
        self.check_forward_ones(self.one)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))
        self.check_forward_ones(cuda.to_gpu(self.one))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), False)
        self.check_forward_ones(cuda.to_gpu(self.one), False)

    def check_backward(self, x_data, y_grad, use_cudnn=True):
        gradient_check.check_backward(
            functions.SpatialPyramidPooling2D(
                x_data.shape[1:], self.pyramid_height, self.pooling_class,
                use_cudnn=use_cudnn),
            x_data, y_grad, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), False)


class TestValidDtype(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.randn(5, 3, 5, 5)
        self.v = chainer.Variable(self.x.astype(numpy.float32))

    def check_valid_dtype(self):
        functions.spatial_pyramid_pooling_2d(
            self.v, 3, functions.MaxPooling2D)

    def test_valid_dtype_cpu(self):
        self.check_valid_dtype()

    @attr.gpu
    def test_valid_dtype_gpu(self):
        self.v.to_gpu()
        self.check_valid_dtype()


class TestInvalidDtype(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.randn(5, 3, 5, 5)
        self.v = chainer.Variable(self.x.astype(numpy.int32))

    def check_invalid_dtype(self):
        functions.spatial_pyramid_pooling_2d(
            self.v, 3, functions.MaxPooling2D)

    def test_invalid_dtype_cpu(self):
        with self.assertRaises(type_check.InvalidType):
            self.check_invalid_dtype()

    @attr.gpu
    def test_invalid_dtype_gpu(self):
        self.v.to_gpu()
        with self.assertRaises(type_check.InvalidType):
            self.check_invalid_dtype()


testing.run_module(__name__, __file__)
