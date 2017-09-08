import unittest

import mock
import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'cover_all': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestMaxPooling2D(unittest.TestCase):

    def setUp(self):
        # Avoid unstability of numerical gradient
        self.x = numpy.arange(
            2 * 3 * 4 * 3, dtype=self.dtype).reshape(2, 3, 4, 3)
        numpy.random.shuffle(self.x)
        self.x = 2 * self.x / self.x.size - 1
        if self.cover_all:
            self.gy = numpy.random.uniform(
                -1, 1, (2, 3, 3, 2)).astype(self.dtype)
        else:
            self.gy = numpy.random.uniform(
                -1, 1, (2, 3, 2, 2)).astype(self.dtype)
        self.ggx = numpy.random.uniform(
            -1, 1, (2, 3, 4, 3)).astype(self.dtype)

    def check_forward(self, x_data, use_cudnn='always'):
        x = chainer.Variable(x_data)
        with chainer.using_config('use_cudnn', use_cudnn):
            y = functions.max_pooling_2d(x, 3, stride=2, pad=1,
                                         cover_all=self.cover_all)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        for k in six.moves.range(2):
            for c in six.moves.range(3):
                x = self.x[k, c]
                if self.cover_all:
                    expect = numpy.array([
                        [x[0:2, 0:2].max(), x[0:2, 1:3].max()],
                        [x[1:4, 0:2].max(), x[1:4, 1:3].max()],
                        [x[3:4, 0:2].max(), x[3:4, 1:3].max()]])
                else:
                    expect = numpy.array([
                        [x[0:2, 0:2].max(), x[0:2, 1:3].max()],
                        [x[1:4, 0:2].max(), x[1:4, 1:3].max()]])
                testing.assert_allclose(expect, y_data[k, c])

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    def test_forward_cpu_wide(self):  # see #120
        x_data = numpy.random.rand(2, 3, 15, 15).astype(self.dtype)
        x = chainer.Variable(x_data)
        functions.max_pooling_2d(x, 6, stride=6, pad=0)

    def test_forward_output_size_zero_cpu(self):
        with six.assertRaisesRegex(
                self, AssertionError,
                'Height in the output should be positive.'):
            x_data = numpy.random.rand(4, 4, 1, 4).astype(self.dtype)
            x = chainer.Variable(x_data)
            functions.max_pooling_2d(x, 3, stride=2)
        with six.assertRaisesRegex(
                self, AssertionError,
                'Width in the output should be positive.'):
            x_data = numpy.random.rand(4, 4, 4, 1).astype(self.dtype)
            x = chainer.Variable(x_data)
            functions.max_pooling_2d(x, 3, stride=2)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_non_contiguous(self):
        self.check_forward(cuda.cupy.asfortranarray(cuda.to_gpu(self.x)))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), 'never')

    @attr.gpu
    def test_forward_output_size_zero_gpu(self):
        with six.assertRaisesRegex(
                self, AssertionError,
                'Height in the output should be positive.'):
            x_data = cuda.cupy.random.rand(4, 4, 1, 4).astype(self.dtype)
            x = chainer.Variable(x_data)
            with chainer.using_config('use_cudnn', 'never'):
                functions.max_pooling_2d(x, 3, stride=2)
        with six.assertRaisesRegex(
                self, AssertionError,
                'Width in the output should be positive.'):
            x_data = cuda.cupy.random.rand(4, 4, 4, 1).astype(self.dtype)
            x = chainer.Variable(x_data)
            with chainer.using_config('use_cudnn', 'never'):
                functions.max_pooling_2d(x, 3, stride=2)

    @attr.cudnn
    def test_forward_output_size_zero_cudnn(self):
        with six.assertRaisesRegex(
                self, AssertionError,
                'Height in the output should be positive.'):
            x_data = cuda.cupy.random.rand(4, 4, 1, 4).astype(self.dtype)
            x = chainer.Variable(x_data)
            with chainer.using_config('use_cudnn', 'always'):
                functions.max_pooling_2d(x, 3, stride=2)
        with six.assertRaisesRegex(
                self, AssertionError,
                'Width in the output should be positive.'):
            x_data = cuda.cupy.random.rand(4, 4, 4, 1).astype(self.dtype)
            x = chainer.Variable(x_data)
            with chainer.using_config('use_cudnn', 'always'):
                functions.max_pooling_2d(x, 3, stride=2)

    def check_backward(self, x_data, y_grad, use_cudnn='always'):
        def f(x):
            return functions.max_pooling_2d(
                x, 3, stride=2, pad=1, cover_all=self.cover_all)
        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_backward(
                f, x_data, y_grad, dtype='d', atol=1e-4, rtol=1e-3)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_non_contiguous(self):
        self.check_backward(
            cuda.cupy.asfortranarray(cuda.to_gpu(self.x)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), 'never')

    def test_backward_cpu_more_than_once(self):
        func = functions.MaxPooling2D(
            3, stride=2, pad=1, cover_all=self.cover_all)
        func.apply((self.x,))
        func.backward((0,), (self.gy,))
        func.backward((0,), (self.gy,))

    def check_double_backward(self, x_data, y_grad, x_grad_grad,
                              use_cudnn='always'):
        def f(x):
            y = functions.max_pooling_2d(
                x, 3, stride=2, pad=1, cover_all=self.cover_all)
            return y * y
        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_double_backward(
                f, x_data, y_grad, x_grad_grad,
                dtype='d', atol=1e-4, rtol=1e-3)

    @condition.retry(3)
    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx, 'never')

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_gpu_non_contiguous(self):
        self.check_double_backward(
            cuda.cupy.asfortranarray(cuda.to_gpu(self.x)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.ggx)))

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_gpu_no_cudnn(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx),
            'never')


@testing.parameterize(*testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestMaxPooling2DCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.arange(
            2 * 3 * 4 * 3, dtype=self.dtype).reshape(2, 3, 4, 3)
        self.gy = cuda.cupy.random.uniform(-1, 1,
                                           (2, 3, 2, 2)).astype(self.dtype)

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.max_pooling_2d(
            x, 3, stride=2, pad=1, cover_all=False)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with mock.patch('cupy.cudnn.cudnn.poolingForward') as func:
                self.forward()
                self.assertEqual(func.called,
                                 chainer.should_use_cudnn('>=auto'))

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            expect = chainer.should_use_cudnn('>=auto')
            y = self.forward()
        # should be consistent to forward regardless of use_cudnn config
        y.grad = self.gy
        with mock.patch('cupy.cudnn.cudnn.poolingBackward') as func:
            y.backward()
            self.assertEqual(func.called, expect)


testing.run_module(__name__, __file__)
