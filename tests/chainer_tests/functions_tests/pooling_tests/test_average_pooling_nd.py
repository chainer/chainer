import unittest

import functools
import numpy
import operator
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import conv
from chainer_tests.functions_tests.pooling_tests import pooling_nd_helper


@testing.parameterize(*testing.product({
    'dims': [(4,), (4, 3), (4, 3, 2), (1, 1, 1, 1)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'pad_value': [None, 0],
}))
class TestAveragePoolingND(unittest.TestCase):

    def setUp(self):
        self.ndim = len(self.dims)
        self.ksize = (3,) * self.ndim
        self.stride = (2,) * self.ndim
        self.pad = (1,) * self.ndim

        x_shape = (2, 3) + self.dims
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)

        outs = tuple(conv.get_conv_outsize(d, k, s, p, False)
                     for (d, k, s, p) in six.moves.zip(
                         self.dims, self.ksize, self.stride, self.pad))
        gy_shape = (2, 3) + outs
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)

        self.check_forward_options = {}
        self.check_backward_options = {'eps': 1e-2}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'eps': 1e-2, 'atol': 5e-3, 'rtol': 5e-2}

    def check_forward(self, x_data, use_cudnn='always'):
        dims = self.dims
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        x = chainer.Variable(x_data)
        with chainer.using_config('use_cudnn', use_cudnn):
            y = functions.average_pooling_nd(
                x, ksize, stride, pad, self.pad_value)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        def denom(idx):
            if self.pad_value is None:
                s = 1
                for slic in idx:
                    s *= slic.stop - slic.start
                return s
            else:
                return functools.reduce(operator.mul, ksize)

        self.assertEqual(self.gy.shape, y_data.shape)
        patches = pooling_nd_helper.pooling_patches(
            dims, ksize, stride, pad, False)
        for k in six.moves.range(2):
            for c in six.moves.range(3):
                x = self.x[k, c]
                expect = numpy.array(
                    [x[idx].sum() / denom(idx) for idx in patches])
                expect = expect.reshape(y_data.shape[2:])
                testing.assert_allclose(
                    expect, y_data[k, c], **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.cudnn
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.cudnn
    def test_forward_gpu_non_contiguous(self):
        self.check_forward(cuda.cupy.asfortranarray(cuda.to_gpu(self.x)))

    @attr.gpu
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), 'never')

    @attr.chainerx
    def test_forward_chainerx_native(self):
        self.check_forward(backend.to_chx(self.x), 'never')

    @attr.chainerx
    @attr.gpu
    def test_forward_chainerx_cuda(self):
        self.check_forward(backend.to_chx(cuda.to_gpu(self.x)), 'never')

    def check_forward_consistency_regression(self, x_data, use_cudnn='always'):
        # Regression test to average_pooling_2d.

        if len(self.dims) != 2:
            return

        if self.pad_value != 0:
            # Not supported in average_pooling_2d
            return

        ksize = self.ksize
        stride = self.stride
        pad = self.pad

        with chainer.using_config('use_cudnn', use_cudnn):
            y_nd = functions.average_pooling_nd(
                x_data, ksize, stride=stride, pad=pad,
                pad_value=self.pad_value)
            y_2d = functions.average_pooling_2d(
                x_data, ksize, stride=stride, pad=pad)
        testing.assert_allclose(y_nd.data, y_2d.data)

    def test_forward_consistency_regression_cpu(self):
        self.check_forward_consistency_regression(self.x)

    @attr.cudnn
    def test_forward_consistency_regression_gpu(self):
        self.check_forward_consistency_regression(cuda.to_gpu(self.x))

    @attr.gpu
    def test_forward_consistency_regression_no_cudnn(self):
        self.check_forward_consistency_regression(cuda.to_gpu(self.x), 'never')

    def check_backward(self, x_data, y_grad, use_cudnn='always'):
        def f(x):
            return functions.average_pooling_nd(
                x, self.ksize, stride=self.stride, pad=self.pad)

        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_backward(
                f, x_data, y_grad, dtype=numpy.float64,
                **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.cudnn
    def test_backward_gpu_non_contiguous(self):
        self.check_backward(
            cuda.cupy.asfortranarray(cuda.to_gpu(self.x)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)))

    @attr.gpu
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), 'never')

    @attr.chainerx
    def test_backward_chainerx_native(self):
        self.check_backward(
            backend.to_chx(self.x), backend.to_chx(self.gy), 'never')

    @attr.chainerx
    @attr.gpu
    def test_backward_chainerx_cuda(self):
        def conv(a):
            return backend.to_chx(cuda.to_gpu(a))

        self.check_backward(conv(self.x), conv(self.gy), 'never')

    def check_backward_consistency_regression(self, x_data, gy_data,
                                              use_cudnn='always'):
        # Regression test to two-dimensional average pooling layer.

        if len(self.dims) != 2:
            return

        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        xp = backend.get_array_module(x_data)

        # Backward computation for N-dimensional average pooling layer.
        x_nd = chainer.Variable(xp.array(x_data))
        with chainer.using_config('use_cudnn', use_cudnn):
            y_nd = functions.average_pooling_nd(
                x_nd, ksize, stride=stride, pad=pad)

        y_nd.grad = gy_data
        y_nd.backward()

        # Backward computation for two-dimensional average pooling layer.
        x_2d = chainer.Variable(xp.array(x_data))
        with chainer.using_config('use_cudnn', use_cudnn):
            y_2d = functions.average_pooling_2d(
                x_2d, ksize, stride=stride, pad=pad)

        y_2d.grad = gy_data
        y_2d.backward()

        # Test that the two result gradients are close enough.
        testing.assert_allclose(x_nd.grad, x_2d.grad)

    def test_backward_consistency_regression_cpu(self):
        self.check_backward_consistency_regression(self.x, self.gy)

    @attr.cudnn
    def test_backward_consistency_regression_gpu(self):
        self.check_backward_consistency_regression(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    def test_backward_consistency_regression_no_cudnn(self):
        self.check_backward_consistency_regression(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), use_cudnn='never')

    def check_double_backward(self, x_data, y_grad, x_grad_grad,
                              use_cudnn='always'):
        def f(x):
            return functions.average_pooling_nd(
                x, self.ksize, stride=self.stride, pad=self.pad)
        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_double_backward(
                f, x_data, y_grad, x_grad_grad, **self.check_backward_options)

    @condition.retry(10)
    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx, 'never')

    @attr.cudnn
    @condition.retry(10)
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))

    @attr.cudnn
    @condition.retry(10)
    def test_double_backward_gpu_non_contiguous(self):
        self.check_double_backward(
            cuda.cupy.asfortranarray(cuda.to_gpu(self.x)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.ggx)))

    @attr.gpu
    @condition.retry(10)
    def test_double_backward_gpu_no_cudnn(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx),
            'never')

    @attr.chainerx
    @condition.retry(10)
    def test_double_backward_chainerx_native(self):
        self.check_double_backward(
            backend.to_chx(self.x), backend.to_chx(self.gy),
            backend.to_chx(self.ggx), 'never')

    @attr.chainerx
    @attr.gpu
    @condition.retry(10)
    def test_double_backward_chainerx_cuda(self):
        def conv(a):
            return backend.to_chx(cuda.to_gpu(a))

        self.check_double_backward(
            conv(self.x), conv(self.gy), conv(self.ggx), 'never')


@testing.parameterize(*testing.product({
    'dims': [(4, 3, 2), (3, 2), (2,)],
    'use_cudnn': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestAveragePoolingNDCudnnCall(unittest.TestCase):

    def setUp(self):
        self.ndim = len(self.dims)
        self.ksize = (3,) * self.ndim
        self.stride = (2,) * self.ndim
        self.pad = (1,) * self.ndim
        x_shape = (2, 3) + self.dims
        self.x = cuda.cupy.arange(functools.reduce(operator.mul, x_shape),
                                  dtype=self.dtype).reshape(x_shape)
        gy_shape = (2, 3) + tuple(
            conv.get_conv_outsize(d, k, s, p)
            for (d, k, s, p)
            in six.moves.zip(self.dims, self.ksize, self.stride, self.pad))
        self.gy = cuda.cupy.random.uniform(-1, 1, gy_shape).astype(self.dtype)

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.average_pooling_nd(
            x, self.ksize, self.stride, self.pad)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with testing.patch('cupy.cudnn.pooling_forward') as func:
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
        with testing.patch('cupy.cudnn.pooling_backward') as func:
            y.backward()
            self.assertEqual(func.called, expect)


class TestAveragePoolingNDWrappers(unittest.TestCase):

    def _get_data(self, ndim):
        x_shape = (2, 3) + (3,) * ndim
        dtype = numpy.float32

        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        ksize = (2,) * ndim

        return x, ksize

    def test_average_pooling_1d(self):
        (x, ksize) = self._get_data(1)
        testing.assert_allclose(
            functions.average_pooling_nd(x, ksize).data,
            functions.average_pooling_1d(x, ksize).data)

    def test_average_pooling_1d_invalid(self):
        (x, ksize) = self._get_data(2)
        with self.assertRaises(ValueError):
            functions.average_pooling_1d(x, ksize)

    def test_average_pooling_3d(self):
        (x, ksize) = self._get_data(3)
        testing.assert_allclose(
            functions.average_pooling_nd(x, ksize).data,
            functions.average_pooling_3d(x, ksize).data)

    def test_average_pooling_3d_invalid(self):
        (x, ksize) = self._get_data(2)
        with self.assertRaises(ValueError):
            functions.average_pooling_3d(x, ksize)


testing.run_module(__name__, __file__)
