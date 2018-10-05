import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer.functions.activation import lstm
from chainer import gradient_check
from chainer import testing
from chainer.testing import backend


def _sigmoid(x):
    half = x.dtype.type(0.5)
    return numpy.tanh(x * half) * half + half


def inject_backend_tests(method_names):
    decorator = backend.inject_backend_tests(
        method_names,
        # CPU tests
        testing.product({
            'use_cuda': [False],
            'use_ideep': ['never', 'always'],
        })
        # GPU tests
        + [{'use_cuda': True}])
    return decorator


@testing.parameterize(*(testing.product({
    'batch': [3, 2, 0],
    'dtype': [numpy.float32],
}) + testing.product({
    'batch': [3],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
@testing.fix_random()
@inject_backend_tests([
    'test_forward',
    'test_flat_forward',
    'test_full_backward',
    'test_flat_full_backward',
    'test_no_gc_backward',
    'test_flat_no_gc_backward',
    'test_no_gh_backward',
    'test_flat_no_gh_backward',
    'test_double_backward'])
class TestLSTM(unittest.TestCase):

    def setUp(self):
        dtype = self.dtype
        hidden_shape = (3, 2, 4)
        x_shape = (self.batch, 8, 4)
        y_shape = (self.batch, 2, 4)

        c_prev = numpy.random.uniform(-1, 1, hidden_shape).astype(dtype)
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)

        gc = numpy.random.uniform(-1, 1, hidden_shape).astype(dtype)
        gh = numpy.random.uniform(-1, 1, y_shape).astype(dtype)

        ggc = numpy.random.uniform(-1, 1, hidden_shape).astype(dtype)
        ggx = numpy.random.uniform(-1, 1, x_shape).astype(dtype)

        self.inputs = [c_prev, x]
        self.grad_outputs = [gc, gh]
        self.grad_grad_inputs = [ggc, ggx]

        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        self.check_double_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-3, 'rtol': 5e-2}
            self.check_double_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-3, 'rtol': 5e-2}

    def flat(self, arrays):
        return [None if a is None else a[:, :, 0] for a in arrays]

    def forward_cpu(self, inputs):
        c_prev, x = inputs
        batch = x.shape[0]
        a_in = x[:, [0, 4]]
        i_in = x[:, [1, 5]]
        f_in = x[:, [2, 6]]
        o_in = x[:, [3, 7]]
        c_expect = (_sigmoid(i_in) * numpy.tanh(a_in)
                    + _sigmoid(f_in) * c_prev[:batch])
        h_expect = _sigmoid(o_in) * numpy.tanh(c_expect)
        return c_expect, h_expect

    def check_forward(self, inputs, backend_config):
        # Compute expected out
        c_prev, x = inputs
        batch = x.shape[0]
        c_expect_2 = c_prev[batch:]
        c_expect_1, h_expect = self.forward_cpu(inputs)

        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
        inputs = [chainer.Variable(xx) for xx in inputs]

        with backend_config:
            c, h = functions.lstm(*inputs)
            assert c.data.dtype == self.dtype
            assert h.data.dtype == self.dtype

        testing.assert_allclose(
            c_expect_1, c.data[:batch], **self.check_forward_options)
        testing.assert_allclose(
            c_expect_2, c.data[batch:], **self.check_forward_options)
        testing.assert_allclose(
            h_expect, h.data, **self.check_forward_options)

    def test_forward(self, backend_config):
        self.check_forward(self.inputs, backend_config)

    def test_flat_forward(self, backend_config):
        self.check_forward(self.flat(self.inputs), backend_config)

    def check_backward(self, inputs, grad_outputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)

        with backend_config:
            gradient_check.check_backward(
                functions.lstm, inputs, grad_outputs,
                **self.check_backward_options)

    def test_full_backward(self, backend_config):
        self.check_backward(self.inputs, self.grad_outputs, backend_config)

    def test_flat_full_backward(self, backend_config):
        self.check_backward(
            self.flat(self.inputs), self.flat(self.grad_outputs),
            backend_config)

    def test_no_gc_backward(self, backend_config):
        grad_outputs = [None, self.grad_outputs[1]]
        self.check_backward(self.inputs, grad_outputs, backend_config)

    def test_flat_no_gc_backward(self, backend_config):
        grad_outputs = [None, self.grad_outputs[1]]
        self.check_backward(
            self.flat(self.inputs), self.flat(grad_outputs), backend_config)

    def test_no_gh_backward(self, backend_config):
        grad_outputs = [self.grad_outputs[0], None]
        self.check_backward(self.inputs, grad_outputs, backend_config)

    def test_flat_no_gh_backward(self, backend_config):
        grad_outputs = [self.grad_outputs[0], None]
        self.check_backward(
            self.flat(self.inputs), self.flat(grad_outputs), backend_config)

    def check_double_backward(
            self, inputs, grad_outputs, grad_grad_inputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)
            grad_grad_inputs = cuda.to_gpu(grad_grad_inputs)

        with backend_config:
            gradient_check.check_double_backward(
                chainer.functions.lstm, inputs, grad_outputs, grad_grad_inputs,
                **self.check_double_backward_options)

    def test_double_backward(self, backend_config):
        self.check_double_backward(
            self.inputs, self.grad_outputs, self.grad_grad_inputs,
            backend_config)


@testing.parameterize(*(testing.product({
    'batch': [3, 2, 0],
    'dtype': [numpy.float32],
}) + testing.product({
    'batch': [3],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
@testing.fix_random()
@inject_backend_tests(['test_backward'])
class TestLSTMGrad(unittest.TestCase):

    def setUp(self):
        hidden_shape = (3, 2, 4)
        dtype = self.dtype
        x_shape = (self.batch, 8, 4)
        y_shape = (self.batch, 2, 4)

        c_prev = numpy.random.uniform(
            -1, 1, hidden_shape).astype(dtype)
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        c_next = numpy.random.uniform(-1, 1, hidden_shape).astype(dtype)

        gc = numpy.random.uniform(-1, 1, hidden_shape).astype(dtype)
        gh = numpy.random.uniform(-1, 1, y_shape).astype(dtype)

        ggc_prev = numpy.random.uniform(-1, 1, hidden_shape).astype(dtype)
        ggx = numpy.random.uniform(-1, 1, x_shape).astype(dtype)

        self.inputs = [c_prev, x, c_next, gc, gh]
        self.grad_outputs = [ggc_prev, ggx]

        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-3, 'rtol': 1e-2}

    def check_backward(self, inputs, grad_outputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)

        with backend_config:
            gradient_check.check_backward(
                lstm.LSTMGrad(), inputs, grad_outputs,
                **self.check_backward_options)

    def test_backward(self, backend_config):
        self.check_backward(self.inputs, self.grad_outputs, backend_config)


testing.run_module(__name__, __file__)
