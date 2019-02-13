import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
import chainerx


@testing.parameterize(*testing.product_dict(
    testing.product({
        'dtype': [numpy.float16, numpy.float32, numpy.float64],
    }),
    testing.product({
        'shape': [None, (2, 3), (2, 2, 3), (2, 2, 2, 3)],
        'axis': [1],
    }) + [
        {'shape': (2, 3), 'axis': 0},
        {'shape': (2, 2, 3), 'axis': -1},
        {'shape': (2, 2, 2, 3), 'axis': -4},
    ],
))
@testing.fix_random()
class TestLogSoftmax(unittest.TestCase):

    def setUp(self):
        if self.shape is None:
            # For checking numerical stability
            value = -5 if self.dtype == numpy.float16 else -1000
            self.x = numpy.array([[value, 1]], dtype=self.dtype)
        else:
            self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.x.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.x.shape).astype(self.dtype)

        self.check_forward_options = {}
        self.check_backward_options = {}
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-3, 'rtol': 5e-2}
            self.check_backward_options = {'atol': 5e-3, 'rtol': 5e-2}
            self.check_double_backward_options = {'atol': 1e-2, 'rtol': 1e-1}

    def check_forward(self, x_data, use_cudnn='always'):
        x = chainer.Variable(x_data)
        with chainer.using_config('use_cudnn', use_cudnn):
            y = functions.log_softmax(x, axis=self.axis)
        self.assertEqual(y.data.dtype, self.dtype)

        log_z = numpy.ufunc.reduce(
            numpy.logaddexp, self.x, axis=self.axis, keepdims=True)
        y_expect = self.x - log_z

        testing.assert_allclose(
            y_expect, y.data, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    def test_forward_gpu_non_contiguous(self):
        self.check_forward(
            cuda.cupy.asfortranarray(cuda.to_gpu(self.x)))

    @attr.gpu
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), 'never')

    @attr.chainerx
    def test_forward_chainerx(self):
        self.check_forward(chainerx.array(self.x))

    def check_backward(self, x_data, gy_data, use_cudnn='always'):
        def f(x):
            return functions.log_softmax(x, self.axis)

        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_backward(
                f, x_data, gy_data, dtype=numpy.float64,
                **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    def test_backward_gpu_non_contiguous(self):
        self.check_backward(
            cuda.cupy.asfortranarray(cuda.to_gpu(self.x)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)))

    @attr.gpu
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), 'never')

    @attr.chainerx
    def test_backward_chainerx(self):
        self.check_backward(chainerx.array(self.x), chainerx.array(self.gy))

    def check_double_backward(self, x_data, gy_data, ggx_data,
                              use_cudnn='always'):
        def f(x):
            return functions.log_softmax(x, self.axis)

        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_double_backward(
                f, x_data, gy_data, ggx_data,
                dtype=numpy.float64, **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))

    @attr.gpu
    def test_double_backward_gpu_no_cudnn(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx),
            'never')

    @attr.chainerx
    def test_double_backward_chainerx(self):
        self.check_double_backward(
            chainerx.array(self.x),
            chainerx.array(self.gy),
            chainerx.array(self.ggx))


@testing.parameterize(*testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestLogSoftmaxCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.expect = chainer.should_use_cudnn('>=auto')

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.log_softmax(x)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with testing.patch('cupy.cudnn.softmax_forward') as func:
                self.forward()
                self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            y.grad = self.gy
            with testing.patch('cupy.cudnn.softmax_backward') as func:
                y.backward()
                self.assertEqual(func.called, self.expect)


testing.run_module(__name__, __file__)
