import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend


def _to_noncontiguous(arrays):
    xp = chainer.backend.get_array_module(*arrays)
    return [xp.asfortranarray(a) for a in arrays]


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'c_contiguous': [True, False],
}))
@testing.fix_random()
@backend.inject_backend_tests(
    ['test_forward', 'test_backward', 'test_double_backward'],
    # CPU tests
    testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
    }))
class TestReLU(unittest.TestCase):

    def setUp(self):
        # Avoid unstability of numerical grad
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        x[(-0.1 < x) & (x < 0.1)] = 0.5
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.inputs = [x]
        self.grad_outputs = [gy]
        self.grad_grad_inputs = [ggx]
        self.check_backward_options = {}
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def forward_cpu(self, inputs):
        x, = inputs
        expected = x.copy()
        expected[expected < 0] = 0
        return expected,

    def check_forward(self, inputs, backend_config):
        y_expected, = self.forward_cpu(inputs)

        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
        if not self.c_contiguous:
            inputs = _to_noncontiguous(inputs)

        x_data, = inputs
        x = chainer.Variable(x_data)
        with backend_config:
            y = functions.relu(x)
        assert y.data.dtype == self.dtype

        testing.assert_allclose(y_expected, y.data)

    def test_forward(self, backend_config):
        self.check_forward(self.inputs, backend_config)

    def check_backward(self, inputs, grad_outputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)
        if not self.c_contiguous:
            inputs = _to_noncontiguous(inputs)
            grad_outputs = _to_noncontiguous(grad_outputs)

        with backend_config:
            gradient_check.check_backward(
                functions.relu, inputs, grad_outputs, dtype=numpy.float64,
                **self.check_backward_options)

    def test_backward(self, backend_config):
        self.check_backward(self.inputs, self.grad_outputs, backend_config)

    def check_double_backward(
            self, inputs, grad_outputs, grad_grad_inputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)
            grad_grad_inputs = cuda.to_gpu(grad_grad_inputs)
        if not self.c_contiguous:
            inputs = _to_noncontiguous(inputs)
            grad_outputs = _to_noncontiguous(grad_outputs)
            grad_grad_inputs = _to_noncontiguous(grad_grad_inputs)

        x, = inputs
        gy, = grad_outputs
        ggx, = grad_grad_inputs
        with backend_config:
            gradient_check.check_double_backward(
                functions.relu, x, gy, ggx, dtype=numpy.float64,
                **self.check_double_backward_options)

    def test_double_backward(self, backend_config):
        self.check_double_backward(
            self.inputs, self.grad_outputs, self.grad_grad_inputs,
            backend_config)


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


testing.run_module(__name__, __file__)
