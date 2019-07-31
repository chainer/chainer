import unittest
import six

import numpy

from chainer.backends import cuda
from chainer import gradient_check
import chainer.functions as F
from chainer import testing
from chainer.functions.rnn import lstm
from chainer.testing import backend


def sigmoid(x):
    return numpy.tanh(x * 0.5) * 0.5 + 0.5


def _shaped_random(shape, dtype):
    return numpy.random.uniform(-1, 1, shape).astype(dtype)


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
        {'c_shape': (20, 32, 4), 'x_shape': (16, 128, 4)},
        {'c_shape': (32, 100, 3, 5), 'x_shape': (32, 400, 3, 5)},
        {'c_shape': (16, 20), 'x_shape': (2, 80)},
        {'c_shape': (16, 20), 'x_shape': (0, 80)},
        {'c_shape': (0, 0), 'x_shape': (0, 0)},
        {'c_shape': (8, 0), 'x_shape': (2, 0)},
    ], [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ], [
        {'grad_outputs': (True, True)},
        {'grad_outputs': (True, False)},
        {'grad_outputs': (False, True)},
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
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {'atol': 5e-3, 'rtol': 5e-2}
            self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-2}
        if self.grad_outputs[0] is False or self.grad_outputs[1] is False:
            self.skip_double_backward_test = True

    def generate_inputs(self):
        c = _shaped_random(self.c_shape, self.dtype)
        x = _shaped_random(self.x_shape, self.dtype)
        return c, x,

    def forward(self, inputs, device):
        c, x = inputs
        c, h = F.lstm(c, x)
        return c, h,

    def forward_expected(self, inputs):
        c, x = inputs
        batch = x.shape[0]

        def _extract_gates(x):
            r = x.reshape((len(x), x.shape[1] // 4, 4) + x.shape[2:])
            return [r[:, :, i] for i in six.moves.range(4)]
        a, i, f, o = _extract_gates(x)
        a = numpy.tanh(a)
        i = sigmoid(i)
        f = sigmoid(f)
        o = sigmoid(o)
        c_exp = numpy.zeros_like(c)
        c_exp[:batch] = a * i + f * c[:batch]
        h_exp = o * numpy.tanh(c_exp[:batch])
        c_exp[batch:] = c[batch:]
        return c_exp, h_exp,

    def generate_grad_outputs(self, outputs_template):
        grad_out = []
        c = outputs_template[0]
        h = outputs_template[1]

        c_shape = c.shape
        h_shape = h.shape
        if self.grad_outputs[0] is True:
            grad_out.append(numpy.random.uniform(-1, 1,
                                                 c_shape).astype(c.dtype))
        else:
            grad_out.append(None)

        if self.grad_outputs[1] is True:
            grad_out.append(numpy.random.uniform(-1, 1,
                                                 h_shape).astype(h.dtype))
        else:
            grad_out.append(None)
        return tuple(grad_out)

    def _generate_grad_outputs(self, outputs_template):
        grad_outputs = self.generate_grad_outputs(outputs_template)
        return grad_outputs


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
