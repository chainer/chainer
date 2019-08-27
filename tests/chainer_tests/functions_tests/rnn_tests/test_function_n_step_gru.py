import numpy

import chainer
import chainer.functions as F
from chainer import testing
from chainer.testing import backend


def sigmoid(x):
    return numpy.tanh(x * 0.5) * 0.5 + 0.5


def array(shape, dtype):
    return numpy.random.uniform(-1, 1, shape).astype(dtype)


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
class TestNStepGRU(testing.FunctionTestCase):

    dodge_nondifferentiable = True

    def setUp(self):
        self.skip_double_backward_test = True

    def generate_inputs(self):
        h_shape = (self.n_layers, self.batches[0], self.hidden_size)
        dtype = numpy.float32

        h = array(h_shape, dtype)
        in_size = self.input_size
        out_size = self.hidden_size
        xs = [array((self.batches[b], in_size), dtype)
              for b in range(len(self.batches))]

        def w_in(i, j):
            return in_size if i == 0 and j < 3 else out_size

        inputs = []
        inputs.append(h)
        for i in range(len(self.batches)):
            inputs.append(xs[i])
        for n in range(self.n_layers):
            for i in range(6):
                inputs.append(array((out_size, w_in(n, i)), dtype))
            for i in range(6):
                inputs.append(array((out_size,), dtype))
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
        if h.array.dtype == numpy.float64:
            with chainer.using_config('use_cudnn', 'never'):
                out = F.n_step_gru(self.n_layers, 0.0, h, ws, bs, xs)
        else:
            out = F.n_step_gru(self.n_layers, 0.0, h, ws, bs, xs)

        rets = []
        rets.append(out[0])
        for i in range(len(out[1])):
            rets.append(out[1][i])
        return tuple(rets)

    def forward_expected(self, inputs):
        h, ws, bs, xs = self.process_inputs(inputs)
        e_hy = h.copy()
        ys = []
        for ind in range(len(xs)):
            x = xs[ind]
            batch = x.shape[0]
            for layer in range(self.n_layers):
                w = ws[layer]
                b = bs[layer]
                h_prev = e_hy[layer, :batch]

                # GRU
                z = sigmoid(x.dot(w[1].T) + h_prev.dot(w[4].T) + b[1] + b[4])
                r = sigmoid(x.dot(w[0].T) + h_prev.dot(w[3].T) + b[0] + b[3])
                h_bar = numpy.tanh(x.dot(w[2].T) +
                                   r *
                                   ((h_prev).dot(w[5].T) + b[5]) + b[2])
                e_h = (1 - z) * h_bar + z * h_prev
                e_hy[layer, :batch] = e_h

                x = e_h
            ys.append(x)
        rets = []
        rets.append(e_hy)
        for i in range(len(ys)):
            rets.append(ys[i])
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
class TestNStepBiGRU(testing.FunctionTestCase):
    dodge_nondifferentiable = True

    def setUp(self):
        self.skip_double_backward_test = True

    def generate_inputs(self):
        h_shape = (self.n_layers * 2, self.batches[0], self.hidden_size)
        dtype = numpy.float32

        h = array(h_shape, dtype)
        in_size = self.input_size
        out_size = self.hidden_size
        xs = [array((self.batches[b], in_size), dtype)
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
                    inputs.append(array((out_size, w_in(n, i)), dtype))
                for i in range(6):
                    inputs.append(array((out_size,), dtype))
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
        if h.array.dtype == numpy.float64:
            with chainer.using_config('use_cudnn', 'never'):
                out = F.n_step_bigru(self.n_layers, 0.0, h, ws, bs, xs)
        else:
            out = F.n_step_bigru(self.n_layers, 0.0, h, ws, bs, xs)
        rets = []
        rets.append(out[0])
        for i in range(len(out[1])):
            rets.append(out[1][i])
        return tuple(rets)

    def forward_expected(self, inputs):
        h, ws, bs, xs = self.process_inputs(inputs)
        xs_next = xs
        e_hy = h.copy()
        for layer in range(self.n_layers):
            # forward
            di = 0
            xf = []
            layer_idx = layer * 2 + di
            w = ws[layer_idx]
            b = bs[layer_idx]
            for ind in range(len(xs)):
                x = xs_next[ind]
                batch = x.shape[0]
                h_prev = e_hy[layer_idx, :batch]
                # GRU
                z = sigmoid(x.dot(w[1].T) + h_prev.dot(w[4].T) + b[1] + b[4])
                r = sigmoid(x.dot(w[0].T) + h_prev.dot(w[3].T) + b[0] + b[3])
                h_bar = numpy.tanh(x.dot(w[2].T) +
                                   r *
                                   ((h_prev).dot(w[5].T) + b[5]) + b[2])
                e_h = (1 - z) * h_bar + z * h_prev
                e_hy[layer_idx, :batch] = e_h
                xf.append(e_h)

            # backward
            di = 1
            xb = []
            layer_idx = layer * 2 + di
            w = ws[layer_idx]
            b = bs[layer_idx]
            for ind in reversed(range(len(xs))):
                x = xs_next[ind]
                batch = x.shape[0]
                h_prev = e_hy[layer_idx, :batch]
                # GRU
                z = sigmoid(x.dot(w[1].T) + h_prev.dot(w[4].T) + b[1] + b[4])
                r = sigmoid(x.dot(w[0].T) + h_prev.dot(w[3].T) + b[0] + b[3])
                h_bar = numpy.tanh(x.dot(w[2].T) +
                                   r *
                                   ((h_prev).dot(w[5].T) + b[5]) + b[2])
                e_h = (1 - z) * h_bar + z * h_prev
                e_hy[layer_idx, :batch] = e_h
                xb.append(e_h)
            xb.reverse()
            xs_next = [numpy.concatenate([hfi, hbi], axis=1) for (hfi, hbi) in
                       zip(xf, xb)]

        rets = []
        rets.append(e_hy)
        for x in xs_next:
            rets.append(x)
        return tuple(rets)


testing.run_module(__name__, __file__)
