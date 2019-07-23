import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import gradient_check
import chainer.functions as F
from chainer import testing
from chainer.functions.rnn import lstm
from chainer.testing import attr
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


@testing.parameterize(*testing.product_dict(
    [
        {'c_shape': (10, 3), 'x_shape': (10, 12)},
        {'c_shape': (20, 32), 'x_shape': (16, 128)},
        {'c_shape': (32, 100), 'x_shape': (32, 400)},
        {'c_shape': (16, 20), 'x_shape': (2, 80)},
    ], [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
@testing.fix_random()
@backend.inject_backend_tests(
    None,
    # ChainerX tests
    testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
    # CPU tests
    + testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + testing.product([
        [{'use_cuda': True}],

        # Without cuDNN
        testing.product({
            'use_cudnn': ['never'],
        })
        # With cuDNN
        + testing.product({
            'use_cudnn': ['always'],
            'cudnn_deterministic': [True, False],
            'autotune': [True, False],
})]))
class TestLSTM(testing.FunctionTestCase):

    dodge_nondifferentiable = True
    def setUp(self):
        dtype = self.dtype

        if 0 in self.c_shape or 0 in self.x_shape:
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        if dtype == numpy.float16:
            self.check_forward_options.update({
                'rtol': 1e-2, 'atol': 1e-2})
            self.check_backward_options.update({
                'rtol': 1e-2, 'atol': 1e-2})
            self.check_double_backward_options.update({'rtol': 1e-2, 'atol': 1e-2})
    def generate_inputs(self):
        c = numpy.random.uniform(-1, 1, self.c_shape).astype(self.dtype)
        x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        return c, x,

    def forward(self, inputs, device):
        c, x = inputs
        c, h = F.lstm(c, x)
        return c, h,

    def forward_expected(self, inputs):
        c, x = inputs
        with chainer.using_config('use_ideep', 'never'):
        	c, h = F.lstm(c, x)
        	return c.array, h.array, 

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
