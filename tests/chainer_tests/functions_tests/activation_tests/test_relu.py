import unittest

import mock
import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestReLU(unittest.TestCase):

    def setUp(self):
        # Avoid unstability of numerical grad
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        for i in numpy.ndindex(self.shape):
            if -0.1 < self.x[i] < 0.1:
                self.x[i] = 0.5
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options = {'dtype': numpy.float64}

    def check_forward(self, x_data, use_cudnn='always'):
        x = chainer.Variable(x_data)
        with chainer.using_config('use_cudnn', use_cudnn):
            y = functions.relu(x)
        self.assertEqual(y.data.dtype, self.dtype)

        expected = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] < 0:
                expected[i] = 0

        testing.assert_allclose(
            expected, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), 'never')

    def check_backward(self, x_data, y_grad, use_cudnn='always'):
        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_backward(
                functions.relu, x_data, y_grad,
                **self.check_backward_options)

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
        self.check_backward(cuda.cupy.asfortranarray(cuda.to_gpu(self.x)),
                            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)))

    @attr.gpu
    @condition.retry(3)
    def test_backward_cpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), 'never')

    def check_double_backward(self, x_data, y_grad, x_grad_grad,
                              use_cudnn='always'):
        def f(x):
            x = functions.relu(x)
            return x * x

        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_double_backward(
                f, x_data, y_grad, x_grad_grad,
                **self.check_backward_options)

    @condition.retry(1)
    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    @condition.retry(1)
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x),
                                   cuda.to_gpu(self.gy),
                                   cuda.to_gpu(self.ggx))

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_gpu_non_contiguous(self):
        self.check_double_backward(
            cuda.cupy.asfortranarray(cuda.to_gpu(self.x)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.ggx)))

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_cpu_no_cudnn(self):
        self.check_double_backward(cuda.to_gpu(self.x),
                                   cuda.to_gpu(self.gy),
                                   cuda.to_gpu(self.ggx),
                                   'never')


@testing.parameterize(*testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestReLUCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.expect = chainer.should_use_cudnn('==always')

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.relu(x)

    def test_call_cudnn_forward(self):
        default_func = cuda.cupy.cudnn.activation_forward
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with mock.patch('cupy.cudnn.activation_forward') as func:
                func.side_effect = default_func
                self.forward()
                self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            y.grad = self.gy
            default_func = cuda.cupy.cudnn.activation_backward
            with mock.patch('cupy.cudnn.activation_backward') as func:
                func.side_effect = default_func
                y.backward()
                self.assertEqual(func.called, self.expect)


testing.run_module(__name__, __file__)
