import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer.functions.activation import tanh
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
import chainerx


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@testing.fix_random()
class TestTanh(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-.5, .5, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-.5, .5, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

        self.check_backward_options = {}
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-2}

    def check_forward(self, x_data, use_cudnn='always'):
        x = chainer.Variable(x_data)
        with chainer.using_config('use_cudnn', use_cudnn):
            y = functions.tanh(x)
        self.assertEqual(y.data.dtype, self.dtype)
        y_expect = functions.tanh(chainer.Variable(self.x))
        testing.assert_allclose(y_expect.data, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), 'always')

    @attr.gpu
    def test_forward_gpu_non_contiguous(self):
        self.check_forward(cuda.cupy.asfortranarray(cuda.to_gpu(self.x)),
                           'always')

    @attr.gpu
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), 'never')

    @attr.chainerx
    def test_forward_chainerx(self):
        # TODO(sonots): Support float16
        if self.dtype == numpy.float16:
            raise unittest.SkipTest('ChainerX does not support float16')

        self.check_forward(chainerx.array(self.x))

    def check_backward(self, x_data, gy_data, use_cudnn='always'):
        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_backward(
                functions.tanh, x_data, gy_data, dtype=numpy.float64,
                **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    def test_backward_gpu_non_contiguous(self):
        self.check_backward(cuda.cupy.asfortranarray(cuda.to_gpu(self.x)),
                            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)))

    @attr.gpu
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), 'never')

    def check_double_backward(self, x_data, gy_data, ggx_data):
        gradient_check.check_double_backward(
            chainer.functions.tanh,  x_data, gy_data, ggx_data,
            dtype=numpy.float64, **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))


@testing.parameterize(*testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestTanhCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.expect = chainer.should_use_cudnn('==always')

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.tanh(x)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            default_func = cuda.cupy.cudnn.activation_forward
            with testing.patch('cupy.cudnn.activation_forward') as func:
                func.side_effect = default_func
                self.forward()
                self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            y.grad = self.gy
            default_func = cuda.cupy.cudnn.activation_backward
            with testing.patch('cupy.cudnn.activation_backward') as func:
                func.side_effect = default_func
                y.backward()
                self.assertEqual(func.called, self.expect)


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@testing.fix_random()
class TestTanhGrad(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-.5, .5, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def check_backward(self, x_data, y_data, gy_data, ggx_data):
        def f(y, gy):
            return tanh.TanhGrad(x_data).apply((y, gy))[0]

        gradient_check.check_backward(
            f, (y_data, gy_data), ggx_data, dtype=numpy.float64,
            **self.check_backward_options)

    def test_backward_cpu(self):
        y = numpy.array(numpy.tanh(self.x))
        self.check_backward(self.x, y, self.gy, self.ggx)

    @attr.gpu
    def test_backward_gpu(self):
        y = numpy.array(numpy.tanh(self.x))
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(y), cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx))


testing.run_module(__name__, __file__)
