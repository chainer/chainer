import unittest

import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.functions as functions
from chainer import gradient_check
from chainer import testing
from chainer import Variable
from chainer.testing import attr
from chainer.testing import backend


@testing.parameterize(*testing.product_dict(
    [
        {'n_layers': 1, 'hidden_size': 2,
            'input_size': 1, 'batches': (1, 1, 1)},
        {'n_layers': 2, 'hidden_size': 2,
            'input_size': 3, 'batches': (3, 2, 1)},
        {'n_layers': 4, 'hidden_size': 6,
            'input_size': 3, 'batches': (5, 3, 1)},
        {'n_layers': 5, 'hidden_size': 10,
            'input_size': 6, 'batches': (6, 5, 3)},
    ]))
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

        testing.product({
            'use_cudnn': ['always'],
            'cudnn_deterministic': [True, False],
            'autotune': [True, False],
        })]))
class TestNStepGRU(testing.FunctionTestCase):

    dodge_nondifferentiable = True

    def setUp(self):
        self.check_forward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_backward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_double_backward_options.update({
            'rtol': 5e-3, 'atol': 5e-2})
        self.skip_double_backward_test = True

    def generate_inputs(self):
        h_shape = (self.n_layers, self.batches[0], self.hidden_size)
        dtype = numpy.float32

        h = numpy.random.uniform(-1, 1, h_shape).astype(dtype)
        in_size = self.input_size
        out_size = self.hidden_size
        xs = [numpy.random.uniform(-1, 1, (self.batches[b], in_size)).astype(dtype)
              for b in range(len(self.batches))]

        def w_in(i, j):
            return in_size if i == 0 and j < 3 else out_size

        inputs = []
        inputs.append(h)
        for i in range(len(self.batches)):
            inputs.append(xs[i])
        for n in range(self.n_layers):
            for i in range(6):
                inputs.append(numpy.random.uniform(-1, 1,
                                                   (out_size, w_in(n, i))).astype(dtype))
            for i in range(6):
                inputs.append(numpy.random.uniform(-1, 1,
                                                   (out_size,)).astype(dtype))
        return tuple(inputs)

    def process_inputs(self, inputs):
        h = inputs[0]

        xs = inputs[1:1 + len(self.batches)]
        ws = []
        bs = []
        index = 1 + len(self.batches)
        for n in range(self.n_layers):
            ws.append(inputs[index: index + 6])
            bs.append(inputs[index + 6: index + 12])
            index += 12

        return h, ws, bs, xs

    def forward(self, inputs, device):
        h, ws, bs, xs = self.process_inputs(inputs)
        # For some reason even though only float32 arrays are created in generate_inputs(),
        # arrays coming as input here are of type float32 and float64
        if h.array.dtype == numpy.float64:
            raise unittest.SkipTest('float64 not supported')
        out = F.n_step_gru(self.n_layers, 0.0, h, ws, bs, xs)
        rets = []
        rets.append(out[0])
        for i in range(len(out[1])):
            rets.append(out[1][i])
        return tuple(rets)

    def forward_expected(self, inputs):
        h, ws, bs, xs = self.process_inputs(inputs)
        with chainer.using_config('use_ideep', 'never'):
            out = F.n_step_gru(self.n_layers, 0.0, h, ws, bs, xs)
            rets = []
            rets.append(out[0].array)
            for i in range(len(out[1])):
                rets.append(out[1][i].array)
            return tuple(rets)


@testing.parameterize(*testing.product_dict(
    [
        {'n_layers': 1, 'hidden_size': 2,
            'input_size': 1, 'batches': (1, 1, 1)},
        {'n_layers': 2, 'hidden_size': 2,
            'input_size': 3, 'batches': (3, 2, 1)},
        {'n_layers': 4, 'hidden_size': 6,
            'input_size': 3, 'batches': (5, 3, 1)},
        {'n_layers': 5, 'hidden_size': 10,
            'input_size': 6, 'batches': (6, 5, 3)},
    ]))
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

        testing.product({
            'use_cudnn': ['always'],
            'cudnn_deterministic': [True, False],
            'autotune': [True, False],
        })]))
class TestNStepBiGRU(testing.FunctionTestCase):
    dodge_nondifferentiable = True

    def setUp(self):
        self.check_forward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_backward_options.update({
            'rtol': 1e-2, 'atol': 1e-2})
        self.check_double_backward_options.update({
            'rtol': 5e-3, 'atol': 5e-2})
        self.skip_double_backward_test = True

    def generate_inputs(self):
        h_shape = (self.n_layers * 2, self.batches[0], self.hidden_size)
        dtype = numpy.float32

        h = numpy.random.uniform(-1, 1, h_shape).astype(dtype)
        in_size = self.input_size
        out_size = self.hidden_size
        xs = [numpy.random.uniform(-1, 1, (self.batches[b], in_size)).astype(dtype)
              for b in range(len(self.batches))]

        def w_in(i, j):
            if i == 0 and j < 3:
                return in_size
            elif i > 0 and j < 3:
                return out_size * 2
            else:
                return out_size

        inputs = []
        inputs.append(h)
        for i in range(len(self.batches)):
            inputs.append(xs[i])

        for n in range(self.n_layers):
            for direction in (0, 1):
                for i in range(6):
                    inputs.append(numpy.random.uniform(-1, 1,
                                                       (out_size, w_in(n, i))).astype(dtype))
                for i in range(6):
                    inputs.append(numpy.random.uniform(-1, 1,
                                                       (out_size,)).astype(dtype))
        return tuple(inputs)

    def process_inputs(self, inputs):
        h = inputs[0]
        xs = inputs[1:1 + len(self.batches)]
        ws = []
        bs = []
        index = 1 + len(self.batches)
        for n in range(self.n_layers):
            ws.append(inputs[index: index + 6])
            bs.append(inputs[index + 6: index + 12])
            ws.append(inputs[index + 12: index + 18])
            bs.append(inputs[index + 18: index + 24])
            index += 24
        return h, ws, bs, xs

    def forward(self, inputs, device):
        h, ws, bs, xs = self.process_inputs(inputs)
        # For some reason even though only float32 arrays are created in generate_inputs(),
        # arrays coming as input here are of type float32 and float64
        if h.array.dtype == numpy.float64:
            raise unittest.SkipTest('float64 not supported')
        out = F.n_step_bigru(self.n_layers, 0.0, h, ws, bs, xs)
        rets = []
        rets.append(out[0][0])
        for i in range(len(out[1])):
            rets.append(out[1][i])
        return tuple(rets)

    def forward_expected(self, inputs):
        h, ws, bs, xs = self.process_inputs(inputs)
        with chainer.using_config('use_ideep', 'never'):
            out = F.n_step_bigru(self.n_layers, 0.0, h, ws, bs, xs)
            rets = []
            rets.append(out[0][0].array)
            for i in range(len(out[1])):
                rets.append(out[1][i].array)
            return tuple(rets)


testing.run_module(__name__, __file__)
