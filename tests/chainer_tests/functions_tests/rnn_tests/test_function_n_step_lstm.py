import unittest
import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend
from chainer.testing import condition


def rand_vector(shape):
    # return cuda.cupy.random.randint(-2, 2, shape).astype('f')
    return cuda.cupy.random.uniform(-1, 1, shape).astype('f')


def sigmoid(x):
    return numpy.tanh(x * 0.5) * 0.5 + 0.5


def array(shape, dtype):
    return numpy.random.uniform(-1, 1, shape).astype(dtype)


def _stack_weight(ws):
    # TODO(unno): Input of the current LSTM implementation is shuffled
    w = F.stack(ws, axis=1)
    shape = w.shape
    return F.reshape(w, (shape[0] * shape[1],) + shape[2:])


def count_close(x, y, atol=1e-4):
    assert x.shape == y.shape
    return int(sum(abs(x - y) / abs(x) < atol))


def lstm_without_dropout(n_layer, dropout, hx, cx, ws, bs, xs):
    xws = [_stack_weight([w[2], w[0], w[1], w[3]]) for w in ws]
    hws = [_stack_weight([w[6], w[4], w[5], w[7]]) for w in ws]
    xbs = [_stack_weight([b[2], b[0], b[1], b[3]]) for b in bs]
    hbs = [_stack_weight([b[6], b[4], b[5], b[7]]) for b in bs]
    xs = [xs[i] for i in range(3)]
    ys = []
    for x in xs:
        cx_next = []
        hx_next = []
        for layer in range(n_layer):
            c = cx[layer]
            h = hx[layer]

            if layer != 0:
                # Only multiply ratio
                x = x * (1 / (1.0 - dropout))
            lstm_in = F.linear(x, xws[layer], xbs[layer]) + \
                F.linear(h, hws[layer], hbs[layer])
            c_new, h_new = F.lstm(c, lstm_in)
            cx_next.append(c_new)
            hx_next.append(h_new)
            x = h_new
        cx = cx_next
        hx = hx_next
        ys.append(x)
    cy = F.stack(cx)
    hy = F.stack(hx)
    return hy, cy, ys


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
class TestNStepLSTM(testing.FunctionTestCase):

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

        h = array(h_shape, dtype)
        c = array(h_shape, dtype)
        in_size = self.input_size
        out_size = self.hidden_size
        xs = []
        for b in range(len(self.batches)):
            xs.append(array((self.batches[b], in_size), dtype))

        def w_in(i, j):
            return in_size if i == 0 and j < 4 else out_size

        inputs = []
        inputs.append(h)
        inputs.append(c)
        for i in range(len(self.batches)):
            inputs.append(xs[i])
        for n in range(self.n_layers):
            for i in range(8):
                inputs.append(array((out_size, w_in(n, i)), dtype))
            for i in range(8):
                inputs.append(array((out_size,), dtype))
        return tuple(inputs)

    def process_inputs(self, inputs):
        h = inputs[0]
        c = inputs[1]
        xs = inputs[2: 2 + len(self.batches)]
        ws = []
        bs = []
        index = 2 + len(self.batches)
        for n in range(self.n_layers):
            ws.append(inputs[index: index + 8])
            bs.append(inputs[index + 8: index + 16])
            index += 16
        return h, c, ws, bs, xs

    def forward(self, inputs, device):
        h, c, ws, bs, xs = self.process_inputs(inputs)
        if h.array.dtype == numpy.float64:
            with chainer.using_config('use_cudnn', 'never'):
                out = F.n_step_lstm(self.n_layers, 0.0, h, c, ws, bs, xs)
        else:
            out = F.n_step_lstm(self.n_layers, 0.0, h, c, ws, bs, xs)
        rets = []
        rets.append(out[0])
        rets.append(out[1])
        for i in range(len(out[2])):
            rets.append(out[2][i])
        return tuple(rets)

    def forward_expected(self, inputs):
        h, c, ws, bs, xs = self.process_inputs(inputs)
        e_hy = h.copy()
        e_cy = c.copy()
        ys = []
        for ind in range(len(xs)):
            x = xs[ind]
            batch = x.shape[0]
            for layer in range(self.n_layers):
                w = ws[layer]
                b = bs[layer]
                h_prev = e_hy[layer, :batch]
                c_prev = e_cy[layer, :batch]
                i = sigmoid(x.dot(w[0].T) + h_prev.dot(w[4].T) + b[0] + b[4])
                f = sigmoid(x.dot(w[1].T) + h_prev.dot(w[5].T) + b[1] + b[5])
                c_bar = numpy.tanh(
                    x.dot(w[2].T) + h_prev.dot(w[6].T) + b[2] + b[6])
                o = sigmoid(x.dot(w[3].T) + h_prev.dot(w[7].T) + b[3] + b[7])
                e_c = (f * c_prev + i * c_bar)
                e_h = o * numpy.tanh(e_c)
                e_hy[layer, :batch] = e_h
                e_cy[layer, :batch] = e_c

                x = e_h
            ys.append(x)
        rets = []
        rets.append(e_hy)
        rets.append(e_cy)
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

        testing.product({
            'use_cudnn': ['always'],
            'cudnn_deterministic': [True, False],
            'autotune': [True, False],
        })]))
class TestNStepBiLSTM(testing.FunctionTestCase):
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

        h = array(h_shape, dtype)
        c = array(h_shape, dtype)
        in_size = self.input_size
        out_size = self.hidden_size
        xs = []
        for b in range(len(self.batches)):
            xs.append(array((self.batches[b], in_size), dtype))

        def w_in(i, j):
            if i == 0 and j < 4:
                return in_size
            elif i > 0 and j < 4:
                return out_size * 2
            else:
                return out_size

        inputs = []
        inputs.append(h)
        inputs.append(c)
        for i in range(len(self.batches)):
            inputs.append(xs[i])
        for n in range(self.n_layers):
            for direction in (0, 1):
                for i in range(8):
                    inputs.append(array((out_size, w_in(n, i)), dtype))
                for i in range(8):
                    inputs.append(array((out_size,), dtype))
        return tuple(inputs)

    def process_inputs(self, inputs):
        h = inputs[0]
        c = inputs[1]
        xs = inputs[2:2 + len(self.batches)]
        ws = []
        bs = []
        index = 2 + len(self.batches)
        for n in range(self.n_layers):
            ws.append(inputs[index: index + 8])
            bs.append(inputs[index + 8: index + 16])
            ws.append(inputs[index + 16: index + 24])
            bs.append(inputs[index + 24: index + 32])
            index += 32
        return h, c, ws, bs, xs

    def forward(self, inputs, device):
        h, c, ws, bs, xs = self.process_inputs(inputs)
        if h.array.dtype == numpy.float64:
            with chainer.using_config('use_cudnn', 'never'):
                out = F.n_step_bilstm(self.n_layers, 0.0, h, c, ws, bs, xs)
        else:
            out = F.n_step_bilstm(self.n_layers, 0.0, h, c, ws, bs, xs)

        rets = []
        rets.append(out[0])
        rets.append(out[1])
        for i in range(len(out[2])):
            rets.append(out[2][i])
        return tuple(rets)

    def forward_expected(self, inputs):
        h, c, ws, bs, xs = self.process_inputs(inputs)
        xs_next = xs
        e_hy = h.copy()
        e_cy = c.copy()
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
                c_prev = e_cy[layer_idx, :batch]
                i = sigmoid(x.dot(w[0].T) + h_prev.dot(w[4].T) + b[0] + b[4])
                f = sigmoid(x.dot(w[1].T) + h_prev.dot(w[5].T) + b[1] + b[5])
                c_bar = numpy.tanh(
                    x.dot(w[2].T) + h_prev.dot(w[6].T) + b[2] + b[6])
                o = sigmoid(x.dot(w[3].T) + h_prev.dot(w[7].T) + b[3] + b[7])
                e_c = (f * c_prev + i * c_bar)
                e_h = o * numpy.tanh(e_c)
                e_hy[layer_idx, :batch] = e_h
                e_cy[layer_idx, :batch] = e_c

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
                c_prev = e_cy[layer_idx, :batch]
                i = sigmoid(x.dot(w[0].T) + h_prev.dot(w[4].T) + b[0] + b[4])
                f = sigmoid(x.dot(w[1].T) + h_prev.dot(w[5].T) + b[1] + b[5])
                c_bar = numpy.tanh(
                    x.dot(w[2].T) + h_prev.dot(w[6].T) + b[2] + b[6])
                o = sigmoid(x.dot(w[3].T) + h_prev.dot(w[7].T) + b[3] + b[7])
                e_c = (f * c_prev + i * c_bar)
                e_h = o * numpy.tanh(e_c)
                e_hy[layer_idx, :batch] = e_h
                e_cy[layer_idx, :batch] = e_c

                xb.append(e_h)

            xb.reverse()
            xs_next = [numpy.concatenate([hfi, hbi], axis=1) for (hfi, hbi) in
                       zip(xf, xb)]

        rets = []
        rets.append(e_hy)
        rets.append(e_cy)
        for x in xs_next:
            rets.append(x)
        return tuple(rets)


@testing.parameterize(*testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
}))
@attr.cudnn
class TestNStepLSTMDropout(unittest.TestCase):

    batch = 20
    length = 3
    in_size = 1
    out_size = 1
    n_layers = 2
    dropout = 0.3
    n_tests = 100

    def setUp(self):
        self.xs = [rand_vector((self.batch, self.in_size))
                   for _ in range(self.length)]
        h_shape = (self.n_layers, self.batch, self.out_size)
        self.cx = rand_vector(h_shape)
        self.hx = rand_vector(h_shape)
        self.ws = []
        self.bs = []
        for i in range(self.n_layers):
            weights = []
            biases = []
            for j in range(8):
                if i == 0 and j < 4:
                    w_in = self.in_size
                else:
                    w_in = self.out_size

                weights.append(rand_vector((self.out_size, w_in)))
                biases.append(rand_vector((self.out_size,)))

            self.ws.append(weights)
            self.bs.append(biases)

    def assert_count(self, actual, expect):
        self.assertTrue(expect * 0.8 < actual < expect * 1.2)

    @condition.retry(5)
    def test_forward_dropout_count(self):
        y_counts = [0] * self.length
        h_counts = [0] * self.n_layers
        c_counts = [0] * self.n_layers

        for _ in range(self.n_tests):
            hy1, cy1, ys1 = lstm_without_dropout(
                self.n_layers, self.dropout, self.hx, self.cx, self.ws,
                self.bs, self.xs)
            with chainer.using_config('use_cudnn', self.use_cudnn):
                hy2, cy2, ys2 = F.n_step_lstm(
                    self.n_layers, self.dropout, self.hx, self.cx, self.ws,
                    self.bs, self.xs)

            for i in range(self.length):
                y_counts[i] += count_close(ys1[i].data, ys2[i].data)

            for i in range(self.n_layers):
                h_counts[i] += count_close(hy1[i].data, hy2[i].data)
                c_counts[i] += count_close(cy1[i].data, cy2[i].data)

        total = self.batch * self.n_tests
        for i in range(self.length):
            self.assert_count(
                y_counts[i],
                total * (1 - self.dropout) ** ((self.n_layers - 1) * (i + 1)))
        for i in range(self.n_layers):
            self.assert_count(
                h_counts[i], total * (1 - self.dropout) ** (self.length * i))
            self.assert_count(
                c_counts[i], total * (1 - self.dropout) ** (self.length * i))


testing.run_module(__name__, __file__)
