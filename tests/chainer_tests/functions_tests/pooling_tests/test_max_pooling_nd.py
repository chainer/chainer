import unittest

import functools
import math
import numpy
from operator import mul
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
import chainerx


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
        self.ggx = numpy.random.uniform(
            -1, 1, x_shape).astype(self.dtype)

        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options = {
                'atol': 1e-3, 'rtol': 1e-2}
            self.check_double_backward_options = {
                'atol': 1e-3, 'rtol': 1e-2}
        else:
            self.check_backward_options = {
                'atol': 1e-4, 'rtol': 1e-3}
            self.check_double_backward_options = {
                'atol': 1e-4, 'rtol': 1e-3}

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
        y_data = backend.to_numpy(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        patches = pooling_nd_helper.pooling_patches(
            dims, ksize, stride, pad, self.cover_all)
        for i in six.moves.range(2):
            for c in six.moves.range(3):
                x = self.x[i, c]
                expect = numpy.array([x[idx].max() for idx in patches])
                expect = expect.reshape(y_data.shape[2:])
                testing.assert_allclose(expect, y_data[i, c])

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

    @attr.chainerx
    def test_forward_chainerx_cpu(self):
        # TODO(sonots): Support it
        if self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')

        self.check_forward(chainerx.array(self.x))

    @attr.chainerx
    @attr.gpu
    @condition.retry(3)
    def test_forward_chainerx_gpu(self):
        # TODO(sonots): Support it
        if self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')

        self.check_forward(backend.to_chainerx(cuda.to_gpu(self.x)))

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

    @attr.chainerx
    @condition.retry(3)
    def test_forward_consistency_regression_chainerx_cpu(self):
        # TODO(sonots): Support it
        if self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')

        self.check_forward_consistency_regression(chainerx.array(self.x))

    @attr.chainerx
    @attr.gpu
    @condition.retry(3)
    def test_forward_consistency_regression_chainerx_gpu(self):
        # TODO(sonots): Support it
        if self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')

        self.check_forward_consistency_regression(
            backend.to_chainerx(cuda.to_gpu(self.x)))

    def check_backward(self, x_data, y_grad, use_cudnn='always'):
        def f(x):
            return functions.max_pooling_nd(
                x, self.ksize, stride=self.stride, pad=self.pad,
                cover_all=self.cover_all)

        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_backward(
                f, x_data, y_grad, dtype='d', **self.check_backward_options)

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

    @attr.chainerx
    @condition.retry(3)
    def test_backward_chainerx_cpu(self):
        # TODO(sonots): Support it
        if self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')

        self.check_backward(chainerx.array(self.x), chainerx.array(self.gy))

    @attr.chainerx
    @attr.gpu
    @condition.retry(3)
    def test_backward_chainerx_gpu(self):
        # TODO(sonots): Support it
        if self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')

        self.check_backward(
            backend.to_chainerx(cuda.to_gpu(self.x)),
            backend.to_chainerx(cuda.to_gpu(self.gy)))

    def check_backward_consistency_regression(self, x_data, gy_data,
                                              use_cudnn='always'):
        # Regression test to two-dimensional max pooling layer.

        if len(self.dims) != 2:
            return

        ksize = self.ksize
        stride = self.stride
        pad = self.pad

        # Backward computation for N-dimensional max pooling layer.
        x_nd = chainer.Variable(x_data.copy())
        with chainer.using_config('use_cudnn', use_cudnn):
            y_nd = functions.max_pooling_nd(
                x_nd, ksize, stride=stride, pad=pad, cover_all=self.cover_all)

        y_nd.grad = gy_data
        y_nd.backward()

        # Backward computation for two-dimensional max pooling layer.
        x_2d = chainer.Variable(x_data.copy())
        with chainer.using_config('use_cudnn', use_cudnn):
            y_2d = functions.max_pooling_2d(
                x_2d, ksize, stride=stride, pad=pad, cover_all=self.cover_all)

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

    @attr.chainerx
    @condition.retry(3)
    def test_backward_consistency_regression_chainerx_cpu(self):
        # TODO(sonots): Support it
        if self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')

        self.check_backward_consistency_regression(
            chainerx.array(self.x), chainerx.array(self.gy))

    @attr.chainerx
    @attr.gpu
    @condition.retry(3)
    def test_backward_consistency_regression_chainerx_gpu(self):
        # TODO(sonots): Support it
        if self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')

        self.check_backward_consistency_regression(
            backend.to_chainerx(cuda.to_gpu(self.x)),
            backend.to_chainerx(cuda.to_gpu(self.gy)))

    def test_backward_cpu_more_than_once(self):
        func = functions.pooling.max_pooling_nd.MaxPoolingND(
            self.ndim, self.ksize, stride=self.stride, pad=self.pad,
            cover_all=self.cover_all)
        func.apply((self.x,))
        func.backward((self.x,), (self.gy,))
        func.backward((self.x,), (self.gy,))

    def check_double_backward(self, x_data, y_grad, x_grad_grad,
                              use_cudnn='always'):
        def f(x):
            return functions.max_pooling_nd(
                x, self.ksize, stride=self.stride, pad=self.pad,
                cover_all=self.cover_all)
        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_double_backward(
                f, x_data, y_grad, x_grad_grad,
                dtype='d',
                **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx, 'never')

    @attr.cudnn
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))

    @attr.cudnn
    def test_double_backward_gpu_non_contiguous(self):
        self.check_double_backward(
            cuda.cupy.asfortranarray(cuda.to_gpu(self.x)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.ggx)))

    @attr.gpu
    def test_double_backward_gpu_no_cudnn(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx),
            'never')

    @attr.chainerx
    def test_double_backward_chainerx_cpu(self):
        # TODO(sonots): Support it
        if self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')

        self.check_double_backward(
            chainerx.array(self.x),
            chainerx.array(self.gy),
            chainerx.array(self.ggx))

    @attr.chainerx
    @attr.gpu
    def test_double_backward_chainerx_gpu(self):
        # TODO(sonots): Support it
        if self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')

        self.check_double_backward(
            backend.to_chainerx(cuda.to_gpu(self.x)),
            backend.to_chainerx(cuda.to_gpu(self.gy)),
            backend.to_chainerx(cuda.to_gpu(self.ggx)))


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
            with testing.patch('cupy.cuda.cudnn.poolingForward') as func:
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
        with testing.patch('cupy.cuda.cudnn.poolingBackward') as func:
            y.backward()
            self.assertEqual(func.called, expect)


class TestMaxPoolingNDWrappers(unittest.TestCase):

    def _get_data(self, ndim):
        x_shape = (2, 3) + (3,) * ndim
        dtype = numpy.float32

        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        ksize = (2,) * ndim

        return x, ksize

    def test_max_pooling_1d(self):
        (x, ksize) = self._get_data(1)
        testing.assert_allclose(
            functions.max_pooling_nd(x, ksize).data,
            functions.max_pooling_1d(x, ksize).data)

    def test_max_pooling_1d_invalid(self):
        (x, ksize) = self._get_data(2)
        with self.assertRaises(ValueError):
            functions.max_pooling_1d(x, ksize)

    def test_max_pooling_3d(self):
        (x, ksize) = self._get_data(3)
        testing.assert_allclose(
            functions.max_pooling_nd(x, ksize).data,
            functions.max_pooling_3d(x, ksize).data)

    def test_max_pooling_3d_invalid(self):
        (x, ksize) = self._get_data(2)
        with self.assertRaises(ValueError):
            functions.max_pooling_3d(x, ksize)


class TestMaxPoolingNDIndices(unittest.TestCase):
    def setUp(self):
        self.x = numpy.arange(
            2 * 3 * 4 * 4, dtype=numpy.float32).reshape(2, 3, 4, 4)

    def _check(self, x):
        out, indices = functions.max_pooling_nd(
            x, 2, cover_all=False, return_indices=True)
        assert isinstance(out, chainer.Variable)
        assert isinstance(out.array, type(x))
        assert isinstance(indices, type(x))
        assert indices.shape == out.array.shape

        # Calculate expected indices.
        expect = numpy.zeros(indices.shape, dtype=indices.dtype)
        for i in six.moves.range(2):
            for c in six.moves.range(3):
                xx = x[i, c]
                expect[i, c] = numpy.array([
                    [xx[0:2, 0:2].ravel().argmax(),
                     xx[0:2, 2:4].ravel().argmax()],
                    [xx[2:4, 0:2].ravel().argmax(),
                     xx[2:4, 2:4].ravel().argmax()],
                ])
        if out.xp is not numpy:
            expect = cuda.to_gpu(expect)
        assert (expect == indices).all()

    def test_cpu(self):
        self._check(self.x)

    @attr.gpu
    @attr.cudnn
    def test_gpu(self):
        x = cuda.to_gpu(self.x)
        with chainer.using_config('use_cudnn', 'never'):
            self._check(x)
        with chainer.using_config('use_cudnn', 'always'):
            self._check(x)


testing.run_module(__name__, __file__)
