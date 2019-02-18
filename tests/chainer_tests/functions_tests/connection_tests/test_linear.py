import unittest

import numpy
import pytest

import chainer
from chainer import functions
from chainer import testing
from chainer.testing import backend


@testing.parameterize(*testing.product({
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'x_shape': [{'n_batch_axes': 1, 'data_shape': (3,)},
                {'n_batch_axes': 3, 'data_shape': (3, 5)},
                ],
    'contiguous': ['C', None],
    'nobias': [True, False],
}))
@backend.inject_backend_tests(
    None,
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
    ]
)
class TestNonparameterizedLinear(testing.FunctionTestCase):

    def setUp(self):
        self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
        self.check_double_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
        if self.x_dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
        self.n_batch_axes = self.x_shape['n_batch_axes']

    def before_test(self, test_name):
        # TODO(crcrpar): Remove this relaxation when
        # a known issue in the reduction of ChainerX is resolved.
        if test_name == 'test_forward':
            if (self.x_dtype == numpy.float16 and
                    self.W_dtype == numpy.float16 and
                    self.n_batch_axes == 3 and
                    self.backend_config.use_chainerx and
                    self.backend_config.chainerx_device == 'native:0'):
                self.check_forward_options['atol'] = 5e-3

    def generate_inputs(self):
        data_shape = self.x_shape['data_shape']
        batch_shape = (4,) + (2,) * (self.n_batch_axes - 1)
        x = numpy.random.uniform(
            -1, 1, batch_shape + data_shape).astype(self.x_dtype)
        input_size = numpy.prod(data_shape)
        W = numpy.random.uniform(-1, 1, (2, input_size)).astype(self.W_dtype)
        if self.nobias:
            return x, W
        else:
            b = numpy.random.uniform(-1, 1, 2).astype(self.x_dtype)
            return x, W, b

    def forward_expected(self, inputs):
        x, W = inputs[:2]
        if self.n_batch_axes > 1:
            batch_shape = x.shape[:self.n_batch_axes]
            batch_size = numpy.prod(batch_shape)
            x = x.reshape(batch_size, -1)
        y = x.dot(W.T)
        if not self.nobias:
            y += inputs[-1]
        if self.n_batch_axes > 1:
            y = y.reshape(batch_shape + (-1,))
        return y.astype(self.x_dtype),

    def forward(self, inputs, device):
        if self.nobias:
            x, W = inputs
            b = None
        else:
            x, W, b = inputs
        y = functions.linear(x, W, b, n_batch_axes=self.n_batch_axes)
        return y,


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
        with pytest.raises(ValueError):
            functions.linear(self.x, self.W, n_batch_axes=n_batch_axes)

    def test_zero(self):
        n_batch_axes = 0
        with pytest.raises(ValueError):
            functions.linear(self.x, self.W, n_batch_axes=n_batch_axes)


testing.run_module(__name__, __file__)
