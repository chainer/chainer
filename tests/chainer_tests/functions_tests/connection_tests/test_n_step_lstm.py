import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def sigmoid(x):
    return numpy.tanh(x * 0.5) * 0.5 + 0.5


def _split(inputs, pos):
    return inputs[:pos], inputs[pos:]


def _to_gpu(x):
    if x is None:
        return None
    elif isinstance(x, list):
        return [_to_gpu(xi) for xi in x]
    else:
        return cuda.to_gpu(x)


def _wrap_variable(x):
    if isinstance(x, list):
        return [_wrap_variable(xi) for xi in x]
    else:
        return chainer.Variable(x)


class TestNStepLSTM(unittest.TestCase):

    batches = [3, 2, 1]
    length = len(batches)
    in_size = 3
    out_size = 2
    n_layers = 2
    dropout = 0.0

    def setUp(self):
        self.xs = [numpy.random.uniform(-1, 1, (b, self.in_size)).astype('f')
                   for b in self.batches]
        h_shape = (self.n_layers, self.batches[0], self.out_size)
        self.cx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)
        self.hx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)

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

                weights.append(numpy.random.uniform(
                    -1, 1, (self.out_size, w_in)).astype('f'))
                biases.append(numpy.random.uniform(
                    -1, 1, (self.out_size,)).astype('f'))
            self.ws.append(weights)
            self.bs.append(biases)

        self.dys = [numpy.random.uniform(-1, 1, (b, self.out_size)).astype('f')
                    for b in self.batches]
        self.dcy = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)
        self.dhy = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)

    def check_forward(
            self, h_data, c_data, xs_data, ws_data, bs_data):
        h = _wrap_variable(h_data)
        c = _wrap_variable(c_data)
        xs = _wrap_variable(xs_data)
        ws = _wrap_variable(ws_data)
        bs = _wrap_variable(bs_data)
        hy, cy, ys = functions.n_step_lstm(
            self.n_layers, self.dropout, h, c, ws, bs, xs)

        e_hy = self.hx.copy()
        e_cy = self.cx.copy()
        for ind in range(self.length):
            x = self.xs[ind]
            batch = x.shape[0]
            for layer in range(self.n_layers):
                w = self.ws[layer]
                b = self.bs[layer]
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

            testing.assert_allclose(
                ys[ind].data, x, rtol=1e-4, atol=1e-4)

        testing.assert_allclose(hy.data, e_hy, rtol=1e-4, atol=1e-4)
        testing.assert_allclose(cy.data, e_cy, rtol=1e-4, atol=1e-4)

    def test_forward_cpu(self):
        self.check_forward(self.hx, self.cx, self.xs, self.ws, self.bs)

    def check_forward_gpu(self, use_cudnn):
        with chainer.using_config('use_cudnn', use_cudnn):
            self.check_forward(
                _to_gpu(self.hx),
                _to_gpu(self.cx),
                _to_gpu(self.xs),
                _to_gpu(self.ws),
                _to_gpu(self.bs))

    @attr.gpu
    def test_forward_gpu_cudnn_always(self):
        self.check_forward_gpu('always')

    @attr.gpu
    def test_forward_gpu_cudnn_auto(self):
        self.check_forward_gpu('auto')

    @attr.gpu
    def test_forward_gpu_cudnn_never(self):
        self.check_forward_gpu('never')

    def check_backward(self, h_data, c_data, xs_data, ws_data, bs_data,
                       dhy_data, dcy_data, dys_data):
        args = tuple([h_data, c_data] + sum(ws_data, []) + sum(bs_data, []) +
                     xs_data)
        grads = tuple([dhy_data, dcy_data] + dys_data)

        def f(*inputs):
            (hx, cx), inputs = _split(inputs, 2)
            ws = []
            for i in range(self.n_layers):
                weights, inputs = _split(inputs, 8)
                ws.append(weights)
            bs = []
            for i in range(self.n_layers):
                biases, inputs = _split(inputs, 8)
                bs.append(biases)
            xs = inputs
            hy, cy, ys = functions.n_step_lstm(
                self.n_layers, self.dropout, hx, cx, ws, bs, xs)
            return (hy, cy) + ys

        gradient_check.check_backward(
            f, args, grads, eps=1e-2, rtol=1e-3, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.hx, self.cx, self.xs, self.ws, self.bs,
                            self.dhy, self.dcy, self.dys)

    @attr.gpu
    def test_backward_gpu(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.check_backward(
                _to_gpu(self.hx),
                _to_gpu(self.cx),
                _to_gpu(self.xs),
                _to_gpu(self.ws),
                _to_gpu(self.bs),
                _to_gpu(self.dhy),
                _to_gpu(self.dcy),
                _to_gpu(self.dys))

    def call_forward(self, train):
        hx = _wrap_variable(_to_gpu(self.hx))
        cx = _wrap_variable(_to_gpu(self.cx))
        xs = _wrap_variable(_to_gpu(self.xs))
        ws = _wrap_variable(_to_gpu(self.ws))
        bs = _wrap_variable(_to_gpu(self.bs))
        with chainer.using_config('enable_backprop', train), \
                chainer.using_config('train', train):
            return functions.n_step_lstm(
                self.n_layers, self.dropout, hx, cx, ws, bs, xs)

    def check_call_cudnn_forward_training(self, use_cudnn):
        with chainer.using_config('use_cudnn', use_cudnn):
            expect = chainer.should_use_cudnn('>=auto', 5000)
            with testing.patch('cupy.cudnn.rnn_forward_training') as func:
                self.call_forward(True)
            assert func.called == expect

    @attr.cudnn
    def test_call_cudnn_forward_training(self):
        self.check_call_cudnn_forward_training('always')
        self.check_call_cudnn_forward_training('never')
        self.check_call_cudnn_forward_training('auto')

    def check_call_cudnn_forward_inference(self, use_cudnn):
        with chainer.using_config('use_cudnn', use_cudnn):
            expect = chainer.should_use_cudnn('>=auto', 5000)
            with testing.patch('cupy.cudnn.rnn_forward_inference') as func:
                self.call_forward(False)
            assert func.called == expect

    @attr.cudnn
    def test_call_cudnn_forward_inference(self):
        self.check_call_cudnn_forward_inference('always')
        self.check_call_cudnn_forward_inference('never')
        self.check_call_cudnn_forward_inference('auto')

    def check_call_cudnn_backward(self, use_cudnn):
        with chainer.using_config('use_cudnn', use_cudnn):
            expect = chainer.should_use_cudnn('>=auto', 5000)
            hy, cy, ys = self.call_forward(True)
            hy.grad = _to_gpu(self.dhy)
            with testing.patch('cupy.cudnn.rnn_backward_weights') as func:
                hy.backward()
            assert func.called == expect

    @attr.cudnn
    def test_call_cudnn_backward(self):
        self.check_call_cudnn_backward('always')
        self.check_call_cudnn_backward('never')
        self.check_call_cudnn_backward('auto')

    def check_inconsistent_input_size(
            self, h_data, c_data, xs_data, ws_data, bs_data):
        h = _wrap_variable(h_data)
        c = _wrap_variable(c_data)
        xs = _wrap_variable(xs_data)
        ws = _wrap_variable(ws_data)
        bs = _wrap_variable(bs_data)
        with self.assertRaises(ValueError):
            functions.n_step_lstm(
                self.n_layers, self.dropout, h, c, ws, bs, xs)

    def test_inconsistent_input_size_cpu(self):
        x_in_size = 4  # inconsistent in_size with that of ws.
        xs = [numpy.random.uniform(-1, 1, (b, x_in_size)).astype('f')
              for b in self.batches]
        self.check_inconsistent_input_size(
            self.hx, self.cx, xs, self.ws, self.bs)

    def check_inconsistent_input_size_gpu(self, use_cudnn):
        x_in_size = 4  # inconsistent in_size with that of ws.
        xs = [numpy.random.uniform(-1, 1, (b, x_in_size)).astype('f')
              for b in self.batches]

        hx = _to_gpu(self.hx)
        cx = _to_gpu(self.cx)
        xs = _to_gpu(xs)
        ws = _to_gpu(self.ws)
        bs = _to_gpu(self.bs)
        with chainer.using_config('use_cudnn', use_cudnn):
            self.check_inconsistent_input_size(hx, cx, xs, ws, bs)

    @attr.gpu
    def test_inconsistent_input_size_gpu_cudnn_always(self):
        self.check_inconsistent_input_size_gpu('always')

    @attr.gpu
    def test_inconsistent_input_size_gpu_cudnn_never(self):
        self.check_inconsistent_input_size_gpu('never')


class TestNStepBiLSTM(unittest.TestCase):

    batches = [3, 2, 1]
    length = len(batches)
    in_size = 3
    out_size = 2
    n_layers = 3
    dropout = 0.0

    def setUp(self):
        self.xs = [numpy.random.uniform(-1, 1, (b, self.in_size)).astype('f')
                   for b in self.batches]
        h_shape = (self.n_layers * 2, self.batches[0], self.out_size)
        self.cx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)
        self.hx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)

        self.ws = []
        self.bs = []
        for i in range(self.n_layers):
            for di in [0, 1]:
                weights = []
                biases = []
                for j in range(8):
                    if i == 0 and j < 4:
                        w_in = self.in_size
                    elif i > 0 and j < 4:
                        w_in = self.out_size * 2
                    else:
                        w_in = self.out_size

                    weights.append(numpy.random.uniform(
                        -1, 1, (self.out_size, w_in)).astype('f'))
                    biases.append(numpy.random.uniform(
                        -1, 1, (self.out_size,)).astype('f'))
                self.ws.append(weights)
                self.bs.append(biases)

        self.dys = [numpy.random.uniform(-1, 1, (b, self.out_size * 2))
                    .astype('f') for b in self.batches]
        self.dcy = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)
        self.dhy = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)

    def check_forward(
            self, h_data, c_data, xs_data, ws_data, bs_data):
        h = _wrap_variable(h_data)
        c = _wrap_variable(c_data)
        xs = _wrap_variable(xs_data)
        ws = _wrap_variable(ws_data)
        bs = _wrap_variable(bs_data)
        hy, cy, ys = functions.n_step_bilstm(
            self.n_layers, self.dropout, h, c, ws, bs, xs)

        xs_next = self.xs
        e_hy = self.hx.copy()
        e_cy = self.cx.copy()
        for layer in range(self.n_layers):
            # forward
            di = 0
            xf = []
            layer_idx = layer * 2 + di
            w = self.ws[layer_idx]
            b = self.bs[layer_idx]
            for ind in range(self.length):
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
            w = self.ws[layer_idx]
            b = self.bs[layer_idx]
            for ind in reversed(range(self.length)):
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

        for k, (ysi, xsi) in enumerate(zip(ys, xs_next)):
            testing.assert_allclose(ysi.data, xsi, rtol=1e-4, atol=1e-4)

        testing.assert_allclose(hy.data, e_hy, rtol=1e-4, atol=1e-4)
        testing.assert_allclose(cy.data, e_cy, rtol=1e-4, atol=1e-4)

    def test_forward_cpu(self):
        self.check_forward(self.hx, self.cx, self.xs, self.ws, self.bs)

    def check_forward_gpu(self, use_cudnn):
        with chainer.using_config('use_cudnn', use_cudnn):
            self.check_forward(
                _to_gpu(self.hx),
                _to_gpu(self.cx),
                _to_gpu(self.xs),
                _to_gpu(self.ws),
                _to_gpu(self.bs))

    @attr.gpu
    def test_forward_gpu_cudnn_always(self):
        self.check_forward_gpu('always')

    @attr.gpu
    def test_forward_gpu_cudnn_auto(self):
        self.check_forward_gpu('auto')

    @attr.gpu
    def test_forward_gpu_cudnn_never(self):
        self.check_forward_gpu('never')

    def check_backward(self, h_data, c_data, xs_data, ws_data, bs_data,
                       dhy_data, dcy_data, dys_data):
        args = tuple([h_data, c_data] + sum(ws_data, []) + sum(bs_data, []) +
                     xs_data)
        grads = tuple([dhy_data, dcy_data] + dys_data)

        def f(*inputs):
            (hx, cx), inputs = _split(inputs, 2)
            ws = []
            for i in range(self.n_layers * 2):
                weights, inputs = _split(inputs, 8)
                ws.append(weights)
            bs = []
            for i in range(self.n_layers * 2):
                biases, inputs = _split(inputs, 8)
                bs.append(biases)
            xs = inputs
            hy, cy, ys = functions.n_step_bilstm(
                self.n_layers, self.dropout, hx, cx, ws, bs, xs)
            return (hy, cy) + ys

        gradient_check.check_backward(
            f, args, grads, eps=1e-2, rtol=1e-3, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.hx, self.cx, self.xs, self.ws, self.bs,
                            self.dhy, self.dcy, self.dys)

    @attr.gpu
    def check_backward_gpu(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.check_backward(
                _to_gpu(self.hx),
                _to_gpu(self.cx),
                _to_gpu(self.xs),
                _to_gpu(self.ws),
                _to_gpu(self.bs),
                _to_gpu(self.dhy),
                _to_gpu(self.dcy),
                _to_gpu(self.dys))

    def call_forward(self, train):
        hx = _wrap_variable(_to_gpu(self.hx))
        cx = _wrap_variable(_to_gpu(self.cx))
        xs = _wrap_variable(_to_gpu(self.xs))
        ws = _wrap_variable(_to_gpu(self.ws))
        bs = _wrap_variable(_to_gpu(self.bs))
        with chainer.using_config('enable_backprop', train), \
                chainer.using_config('train', train):
            return functions.n_step_bilstm(
                self.n_layers, self.dropout, hx, cx, ws, bs, xs)

    def check_call_cudnn_forward_training(self, use_cudnn):
        with chainer.using_config('use_cudnn', use_cudnn):
            expect = chainer.should_use_cudnn('>=auto', 5000)
            with testing.patch('cupy.cudnn.rnn_forward_training') as func:
                self.call_forward(True)
            assert func.called == expect

    @attr.cudnn
    def test_call_cudnn_forward_training(self):
        self.check_call_cudnn_forward_training('always')
        self.check_call_cudnn_forward_training('never')
        self.check_call_cudnn_forward_training('auto')

    def check_call_cudnn_forward_inference(self, use_cudnn):
        with chainer.using_config('use_cudnn', use_cudnn):
            expect = chainer.should_use_cudnn('>=auto', 5000)
            with testing.patch('cupy.cudnn.rnn_forward_inference') as func:
                self.call_forward(False)
            assert func.called == expect

    @attr.cudnn
    def test_call_cudnn_forward_inference(self):
        self.check_call_cudnn_forward_inference('always')
        self.check_call_cudnn_forward_inference('never')
        self.check_call_cudnn_forward_inference('auto')

    def check_call_cudnn_backward(self, use_cudnn):
        with chainer.using_config('use_cudnn', use_cudnn):
            expect = chainer.should_use_cudnn('>=auto', 5000)
            hy, cy, ys = self.call_forward(True)
            hy.grad = _to_gpu(self.dhy)
            with testing.patch('cupy.cudnn.rnn_backward_weights') as func:
                hy.backward()
            assert func.called == expect

    @attr.cudnn
    def test_call_cudnn_backward(self):
        self.check_call_cudnn_backward('always')
        self.check_call_cudnn_backward('never')
        self.check_call_cudnn_backward('auto')

    def check_inconsistent_input_size(
            self, h_data, c_data, xs_data, ws_data, bs_data):
        h = _wrap_variable(h_data)
        c = _wrap_variable(c_data)
        xs = _wrap_variable(xs_data)
        ws = _wrap_variable(ws_data)
        bs = _wrap_variable(bs_data)
        with self.assertRaises(ValueError):
            functions.n_step_bilstm(
                self.n_layers, self.dropout, h, c, ws, bs, xs)

    def test_inconsistent_input_size_cpu(self):
        x_in_size = 4  # inconsistent in_size with that of ws.
        xs = [numpy.random.uniform(-1, 1, (b, x_in_size)).astype('f')
              for b in self.batches]
        self.check_inconsistent_input_size(
            self.hx, self.cx, xs, self.ws, self.bs)

    def check_inconsistent_input_size_gpu(self, use_cudnn):
        x_in_size = 4  # inconsistent in_size with that of ws.
        xs = [numpy.random.uniform(-1, 1, (b, x_in_size)).astype('f')
              for b in self.batches]

        hx = _to_gpu(self.hx)
        cx = _to_gpu(self.cx)
        xs = _to_gpu(xs)
        ws = _to_gpu(self.ws)
        bs = _to_gpu(self.bs)
        with chainer.using_config('use_cudnn', use_cudnn):
            self.check_inconsistent_input_size(hx, cx, xs, ws, bs)

    @attr.gpu
    def test_inconsistent_input_size_gpu_cudnn_always(self):
        self.check_inconsistent_input_size_gpu('always')

    @attr.gpu
    def test_inconsistent_input_size_gpu_cudnn_never(self):
        self.check_inconsistent_input_size_gpu('never')


def _stack_weight(ws):
    # TODO(unno): Input of the current LSTM implementation is shuffled
    w = functions.stack(ws, axis=1)
    shape = w.shape
    return functions.reshape(w, (shape[0] * shape[1],) + shape[2:])


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
            lstm_in = functions.linear(x, xws[layer], xbs[layer]) + \
                functions.linear(h, hws[layer], hbs[layer])
            c_new, h_new = functions.lstm(c, lstm_in)
            cx_next.append(c_new)
            hx_next.append(h_new)
            x = h_new
        cx = cx_next
        hx = hx_next
        ys.append(x)
    cy = functions.stack(cx)
    hy = functions.stack(hx)
    return hy, cy, ys


def rand_vector(shape):
    # return cuda.cupy.random.randint(-2, 2, shape).astype('f')
    return cuda.cupy.random.uniform(-1, 1, shape).astype('f')
    # return cuda.cupy.ones(shape).astype('f')


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
                hy2, cy2, ys2 = functions.n_step_lstm(
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
