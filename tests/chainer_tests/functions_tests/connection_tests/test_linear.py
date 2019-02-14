import unittest

import numpy

import chainer
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import backend
import chainerx


def _to_noncontiguous(arrays):
    xp = chainer.backend.get_array_module(*arrays)
    # TODO(niboshi): Fix it. Non-contiguous tests are skipped for ChainerX.
    if xp is chainerx:
        raise unittest.SkipTest('ChainerX does not support asfortranarray')
    return [None if a is None else xp.asfortranarray(a) for a in arrays]


@testing.parameterize(*testing.product({
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'x_shape': [{'n_batch_axes': 1, 'data_shape': (3,)},
                {'n_batch_axes': 3, 'data_shape': (3, 5)}],
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
    }]
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    ])
class TestNonparameterizedLinear(unittest.TestCase):

    def setUp(self):
        self.n_batch_axes = self.x_shape['n_batch_axes']
        data_shape = self.x_shape['data_shape']
        input_size = numpy.prod(data_shape)
        W = numpy.random.uniform(-1, 1, (2, input_size)).astype(self.W_dtype)
        if self.nobias:
            b = None
        else:
            b = numpy.random.uniform(-1, 1, 2).astype(self.x_dtype)

        batch_shape = (4,) + (2,) * (self.n_batch_axes - 1)
        x = numpy.random.uniform(
            -1, 1, batch_shape + data_shape).astype(self.x_dtype)
        gy = numpy.random.uniform(
            -1, 1, batch_shape + (2,)).astype(self.x_dtype)
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
                'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {
                'atol': 1e-2, 'rtol': 1e-2}

    def forward_cpu(self, inputs):
        x, W, b = inputs
        if self.n_batch_axes > 1:
            batch_shape = x.shape[:self.n_batch_axes]
            batch_size = numpy.prod(batch_shape)
            x = x.reshape(batch_size, -1)
        y = x.dot(W.T)
        if b is not None:
            y += b
        if self.n_batch_axes > 1:
            y = y.reshape(batch_shape + (-1,))
        return y,

    def forward(self, *inputs):
        if len(inputs) == 3:
            x, W, b = inputs
            y = functions.linear(x, W, b, n_batch_axes=self.n_batch_axes)
        else:
            x, W = inputs
            y = functions.linear(x, W, n_batch_axes=self.n_batch_axes)
        return y,

    def test_forward(self, backend_config):
        inputs = self.inputs

        y_expected, = self.forward_cpu(inputs)

        if self.nobias:
            inputs = inputs[:-1]

        inputs = backend_config.get_array(inputs)
        if not self.c_contiguous:
            with backend_config:
                inputs = _to_noncontiguous(inputs)

        input_vars = [chainer.Variable(x) for x in inputs]
        with backend_config:
            y, = self.forward(*input_vars)

        assert y.data.dtype == self.x_dtype
        testing.assert_allclose(
            y_expected, y.data, **self.check_forward_options)

    def test_backward(self, backend_config):
        inputs = self.inputs
        grad_outputs = self.grad_outputs

        if self.nobias:
            inputs = inputs[:-1]

        inputs = backend_config.get_array(inputs)
        grad_outputs = backend_config.get_array(grad_outputs)
        if not self.c_contiguous:
            with backend_config:
                inputs = _to_noncontiguous(inputs)
                grad_outputs = _to_noncontiguous(grad_outputs)

        with backend_config:
            gradient_check.check_backward(
                self.forward, inputs, grad_outputs,
                **self.check_backward_options)

    def test_double_backward(self, backend_config):
        inputs = self.inputs
        grad_outputs = self.grad_outputs
        grad_grad_inputs = self.grad_grad_inputs

        if self.nobias:
            inputs = inputs[:-1]
            grad_grad_inputs = grad_grad_inputs[:-1]

        inputs = backend_config.get_array(inputs)
        grad_outputs = backend_config.get_array(grad_outputs)
        grad_grad_inputs = backend_config.get_array(grad_grad_inputs)

        if not self.c_contiguous:
            with backend_config:
                inputs = _to_noncontiguous(inputs)
                grad_outputs = _to_noncontiguous(grad_outputs)
                grad_grad_inputs = _to_noncontiguous(grad_grad_inputs)

        with backend_config:
            gradient_check.check_double_backward(
                self.forward, inputs, grad_outputs, grad_grad_inputs,
                **self.check_double_backward_options)


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


class TestLinearNBatchAxesBoundaryCondition(unittest.TestCase):

    def setUp(self):
        self.W = numpy.random.uniform(
            -1, 1, (2, 15)).astype(numpy.float32)
        self.x = numpy.random.uniform(
            -1, 1, (3, 3, 5)).astype(numpy.float32)

    def test_negative(self):
        n_batch_axes = -1
        with self.assertRaises(ValueError):
            functions.linear(self.x, self.W, n_batch_axes=n_batch_axes)

    def test_zero(self):
        n_batch_axes = 0
        with self.assertRaises(ValueError):
            functions.linear(self.x, self.W, n_batch_axes=n_batch_axes)


testing.run_module(__name__, __file__)
