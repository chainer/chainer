import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import backend


def _to_noncontiguous(arrays):
    xp = cuda.get_array_module(*arrays)
    return [None if a is None else xp.asfortranarray(a) for a in arrays]


@testing.parameterize(*testing.product({
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'c_contiguous': [True, False],
    'nobias': [True, False],
}))
@backend.inject_backend_tests(
    ['test_forward', 'test_backward', 'test_double_backward'],
    # CPU tests
    testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + [{
        'use_cuda': True,
    }])
class TestNonparameterizedLinear(unittest.TestCase):

    def setUp(self):
        W = numpy.random.uniform(
            -1, 1, (2, 3)).astype(self.W_dtype)
        if self.nobias:
            b = None
        else:
            b = numpy.random.uniform(-1, 1, 2).astype(self.x_dtype)

        x = numpy.random.uniform(-1, 1, (4, 3)).astype(self.x_dtype)
        gy = numpy.random.uniform(-1, 1, (4, 2)).astype(self.x_dtype)
        ggx = numpy.random.uniform(-1, 1, x.shape).astype(self.x_dtype)
        ggW = numpy.random.uniform(-1, 1, W.shape).astype(self.W_dtype)
        if self.nobias:
            ggb = None
        else:
            ggb = numpy.random.uniform(-1, 1, b.shape).astype(self.x_dtype)
        self.inputs = [x, W, b]
        self.grad_outputs = [gy]
        self.grad_grad_inputs = [ggx, ggW, ggb]
        if self.x_dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
        elif self.W_dtype == numpy.float16:
            self.check_forward_options = {}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
        else:
            self.check_forward_options = {}
            self.check_backward_options = {
                'atol': 1e-3, 'rtol': 1e-3}
            self.check_double_backward_options = {
                'atol': 1e-3, 'rtol': 1e-3}

    def forward_cpu(self, inputs):
        x, W, b = inputs
        y = x.dot(W.T)
        if b is not None:
            y += b
        return y,

    def forward(self, *inputs):
        if len(inputs) == 3:
            x, W, b = inputs
            y = functions.linear(x, W, b)
        else:
            x, W = inputs
            y = functions.linear(x, W)
        return y,

    def check_forward(self, inputs, backend_config):
        y_expected, = self.forward_cpu(inputs)

        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
        if not self.c_contiguous:
            inputs = _to_noncontiguous(inputs)

        if self.nobias:
            inputs = inputs[:-1]

        input_vars = [chainer.Variable(x) for x in inputs]
        with backend_config:
            y, = self.forward(*input_vars)

        assert y.data.dtype == self.x_dtype
        testing.assert_allclose(
            y_expected, y.data, **self.check_forward_options)

    def test_forward(self, backend_config):
        self.check_forward(self.inputs, backend_config)

    def check_backward(self, inputs, grad_outputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)
        if not self.c_contiguous:
            inputs = _to_noncontiguous(inputs)
            grad_outputs = _to_noncontiguous(grad_outputs)

        if self.nobias:
            inputs = inputs[:-1]

        with backend_config:
            gradient_check.check_backward(
                self.forward, inputs, grad_outputs,
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

        if self.nobias:
            inputs = inputs[:-1]
            grad_grad_inputs = grad_grad_inputs[:-1]

        # non-linear function for testing
        def nonlinear(*args):
            y, = self.forward(*args)
            return y * y

        with backend_config:
            gradient_check.check_double_backward(
                nonlinear, inputs, grad_outputs, grad_grad_inputs,
                **self.check_double_backward_options)

    def test_double_backward(self, backend_config):
        self.check_double_backward(
            self.inputs, self.grad_outputs, self.grad_grad_inputs,
            backend_config)


class TestLinearBackwardNoncontiguousGradOutputs(unittest.TestCase):
    # NumPy raises an error when the inputs of dot operation are not
    # contiguous. This test ensures this issue is correctly handled.
    # (https://github.com/chainer/chainer/issues/2744)

    # This test depdends on that backward() of F.sum generates
    # a non-contiguous array.

    def test_1(self):
        with chainer.using_config('use_ideep', 'never'):
            n_batches = 1  # important
            in_dims = (2, 2)
            out_dim = 3
            x_shape = (n_batches,) + in_dims
            w_shape = (out_dim, numpy.prod(in_dims),)
            x = numpy.ones(x_shape, numpy.float32)
            w = numpy.ones(w_shape, numpy.float32)
            y = functions.linear(chainer.Variable(x), w)
            z = functions.sum(y)
            z.backward()


testing.run_module(__name__, __file__)
