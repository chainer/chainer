import unittest

import functools
import math
import mock
import numpy
from operator import mul
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import conv
from chainer_tests.functions_tests.pooling_tests import pooling_nd_helper


@testing.parameterize(*testing.product({
    'dims': [(4,), (4, 3), (4, 3, 2), (1, 1, 1, 1)],
    'cover_all': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestMaxPoolingND(unittest.TestCase):

    def setUp(self):
        self.ndim = len(self.dims)
        self.ksize = (3,) * self.ndim
        self.stride = (2,) * self.ndim
        self.pad = (1,) * self.ndim

        # Avoid unstability of numerical gradient
        x_shape = (2, 3) + self.dims
        self.x = numpy.arange(
            functools.reduce(mul, x_shape), dtype=self.dtype).reshape(x_shape)
        self.x = 2 * self.x / self.x.size - 1

        outs = tuple(conv.get_conv_outsize(d, k, s, p, self.cover_all)
                     for (d, k, s, p)
                     in six.moves.zip(
                         self.dims, self.ksize, self.stride, self.pad))
        gy_shape = (2, 3) + outs
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(self.dtype)

        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options = {
                'atol': 1e-03, 'rtol': 1e-03}

    def check_forward(self, x_data, use_cudnn='always'):
        dims = self.dims
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        x = chainer.Variable(x_data)
        with chainer.using_config('use_cudnn', use_cudnn):
            y = functions.max_pooling_nd(x, ksize, stride=stride, pad=pad,
                                         cover_all=self.cover_all)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        patches = pooling_nd_helper.pooling_patches(
            dims, ksize, stride, pad, self.cover_all)
        for k in six.moves.range(2):
            for c in six.moves.range(3):
                x = self.x[k, c]
                expect = numpy.array([x[idx].max() for idx in patches])
                expect = expect.reshape(y_data.shape[2:])
                testing.assert_allclose(expect, y_data[k, c])

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, use_cudnn='never')

    def test_forward_cpu_wide(self):  # see #120
        ndim = self.ndim
        x_shape = (2, 3) + (15,) * ndim
        x_data = numpy.random.rand(*x_shape).astype(self.dtype)
        x = chainer.Variable(x_data)
        ksize = stride = int(math.ceil(pow(32, 1.0 / ndim)))
        functions.max_pooling_nd(x, ksize, stride=stride, pad=0)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu_non_contiguous(self):
        self.check_forward(cuda.cupy.asfortranarray(cuda.to_gpu(self.x)))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), 'never')

    def check_forward_consistency_regression(self, x_data, use_cudnn='always'):
        # Regression test to max_pooling_2d.

        if len(self.dims) != 2:
            return

        ksize = self.ksize
        stride = self.stride
        pad = self.pad

        with chainer.using_config('use_cudnn', use_cudnn):
            y_nd = functions.max_pooling_nd(self.x, ksize, stride=stride,
                                            pad=pad, cover_all=self.cover_all)
            y_2d = functions.max_pooling_2d(self.x, ksize, stride=stride,
                                            pad=pad, cover_all=self.cover_all)
        testing.assert_allclose(y_nd.data, y_2d.data)

    @condition.retry(3)
    def test_forward_consistency_regression_cpu(self):
        self.check_forward_consistency_regression(self.x)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_consistency_regression_gpu(self):
        self.check_forward_consistency_regression(cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_forward_consistency_regression_no_cudnn(self):
        self.check_forward_consistency_regression(cuda.to_gpu(self.x), 'never')

    def check_backward(self, x_data, y_grad, use_cudnn='always'):
        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_backward(
                functions.MaxPoolingND(
                    self.ndim, self.ksize, stride=self.stride, pad=self.pad,
                    cover_all=self.cover_all),
                x_data, y_grad, dtype='d', **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu_non_contiguous(self):
        self.check_backward(
            cuda.cupy.asfortranarray(cuda.to_gpu(self.x)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), 'never')

    def check_backward_consistency_regression(self, x_data, gy_data,
                                              use_cudnn='always'):
        # Regression test to two-dimensional max pooling layer.

        if len(self.dims) != 2:
            return

        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        xp = cuda.get_array_module(x_data)

        # Backward computation for N-dimensional max pooling layer.
        x_nd = chainer.Variable(xp.array(x_data))
        with chainer.using_config('use_cudnn', use_cudnn):
            func_nd = functions.MaxPoolingND(self.ndim, ksize, stride=stride,
                                             pad=pad, cover_all=self.cover_all)
        y_nd = func_nd(x_nd)
        y_nd.grad = gy_data
        y_nd.backward()

        # Backward computation for two-dimensional max pooling layer.
        x_2d = chainer.Variable(xp.array(x_data))
        with chainer.using_config('use_cudnn', use_cudnn):
            func_2d = functions.MaxPooling2D(ksize, stride=stride, pad=pad,
                                             cover_all=self.cover_all)
        y_2d = func_2d.apply((x_2d,))[0]
        y_2d.grad = gy_data
        y_2d.backward()

        # Test that the two result gradients are close enough.
        testing.assert_allclose(x_nd.grad, x_2d.grad)

    @condition.retry(3)
    def test_backward_consistency_regression_cpu(self):
        self.check_backward_consistency_regression(self.x, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_consistency_regression_gpu(self):
        self.check_backward_consistency_regression(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_consistency_regression_no_cudnn(self):
        self.check_backward_consistency_regression(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), use_cudnn='never')

    def test_backward_cpu_more_than_once(self):
        func = functions.MaxPoolingND(
            self.ndim, self.ksize, stride=self.stride, pad=self.pad,
            cover_all=self.cover_all)
        func(self.x)
        func.backward_cpu((self.x,), (self.gy,))
        func.backward_cpu((self.x,), (self.gy,))


@testing.parameterize(*testing.product({
    'dims': [(4, 3, 2), (3, 2), (2,)],
    'use_cudnn': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestMaxPoolingNDCudnnCall(unittest.TestCase):

    def setUp(self):
        self.ndim = len(self.dims)
        self.ksize = (3,) * self.ndim
        self.stride = (2,) * self.ndim
        self.pad = (1,) * self.ndim
        x_shape = (2, 3) + self.dims
        self.x = cuda.cupy.arange(functools.reduce(mul, x_shape),
                                  dtype=self.dtype).reshape(x_shape)
        gy_shape = (2, 3) + tuple(
            conv.get_conv_outsize(d, k, s, p)
            for (d, k, s, p)
            in six.moves.zip(self.dims, self.ksize, self.stride, self.pad))
        self.gy = cuda.cupy.random.uniform(-1, 1, gy_shape).astype(self.dtype)

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.max_pooling_nd(
            x, self.ksize, self.stride, self.pad, cover_all=False)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with mock.patch('cupy.cuda.cudnn.poolingForward') as func:
                self.forward()
                self.assertEqual(func.called,
                                 chainer.should_use_cudnn('>=auto') and
                                 self.ndim > 1)

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            expect = chainer.should_use_cudnn('>=auto') and self.ndim > 1
            y = self.forward()
        # should be consistent to forward regardless of use_cudnn config
        y.grad = self.gy
        with mock.patch('cupy.cuda.cudnn.poolingBackward') as func:
            y.backward()
            self.assertEqual(func.called, expect)


testing.run_module(__name__, __file__)
