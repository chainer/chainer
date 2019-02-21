import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _split(inputs, pos):
    return inputs[:pos], inputs[pos:]


def _relu(x):
    expected = x.copy()
    for i in numpy.ndindex(x.shape):
        if x[i] < 0:
            expected[i] = 0
    return expected


def _to_gpu(x):
    if x is None:
        return None
    elif isinstance(x, list):
        return [_to_gpu(xi) for xi in x]
    else:
        return cuda.to_gpu(x)


def _shaped_random(shape, dtype='f'):
    if isinstance(shape, list):
        return [_shaped_random(s) for s in shape]
    else:
        return numpy.random.uniform(-1, 1, shape).astype(dtype)


def _wrap_variable(x):
    if isinstance(x, list):
        return [_wrap_variable(xi) for xi in x]
    else:
        return chainer.Variable(x)


@testing.parameterize(*testing.product({
    'activation': ['tanh', 'relu']
}))
class TestNStepRNN(unittest.TestCase):

    batches = [3, 2, 1]
    length = len(batches)
    in_size = 3
    out_size = 2
    n_layers = 2
    dropout = 0.0

    def setUp(self):
        self.xs = _shaped_random([(b, self.in_size) for b in self.batches])
        h_shape = (self.n_layers, self.batches[0], self.out_size)
        self.hx = _shaped_random(h_shape)

        o = self.out_size
        i = self.in_size
        self.ws = []
        self.bs = []
        # The first layer has the different shape
        self.ws.append(_shaped_random([(o, i), (o, o)]))
        self.bs.append(_shaped_random([o, o]))
        for _ in range(self.n_layers - 1):
            self.ws.append(_shaped_random([(o, o), (o, o)]))
            self.bs.append(_shaped_random([o, o]))

        self.dys = _shaped_random([(b, self.out_size) for b in self.batches])
        self.dhy = _shaped_random(h_shape)

    def check_forward(
            self, h_data, xs_data, ws_data, bs_data):
        h = _wrap_variable(h_data)
        xs = _wrap_variable(xs_data)
        ws = _wrap_variable(ws_data)
        bs = _wrap_variable(bs_data)
        hy, ys = functions.n_step_rnn(
            self.n_layers, self.dropout, h, ws, bs, xs,
            activation=self.activation)

        e_hy = self.hx.copy()
        for ind in range(self.length):
            x = self.xs[ind]
            batch = x.shape[0]
            for layer in range(self.n_layers):
                w = self.ws[layer]
                b = self.bs[layer]
                h_prev = e_hy[layer, :batch]
                if self.activation == 'tanh':
                    e_h = numpy.tanh(x.dot(w[0].T) +
                                     h_prev.dot(w[1].T) + b[0] + b[1])
                elif self.activation == 'relu':
                    e_h = _relu(x.dot(w[0].T) +
                                h_prev.dot(w[1].T) + b[0] + b[1])

                e_hy[layer, :batch] = e_h

                x = e_h

            testing.assert_allclose(
                ys[ind].data, x, rtol=1e-4, atol=1e-4)

        testing.assert_allclose(hy.data, e_hy, rtol=1e-4, atol=1e-4)

    def test_forward_cpu(self):
        self.check_forward(self.hx, self.xs, self.ws, self.bs)

    def check_forward_gpu(self, use_cudnn):
        with chainer.using_config('use_cudnn', use_cudnn):
            self.check_forward(
                _to_gpu(self.hx),
                _to_gpu(self.xs),
                _to_gpu(self.ws),
                _to_gpu(self.bs))

    @attr.gpu
    def test_forward_gpu_cudnn_always(self):
        self.check_forward_gpu('always')

    @attr.gpu
    def test_forward_gpu_cudnn_never(self):
        self.check_forward_gpu('never')

    def check_backward(self, h_data, xs_data, ws_data, bs_data,
                       dhy_data, dys_data):
        args = tuple([h_data, ] + sum(ws_data, []) + sum(bs_data, []) +
                     xs_data)
        grads = tuple([dhy_data, ] + dys_data)

        def f(*inputs):
            (hx, ), inputs = _split(inputs, 1)
            ws = []
            for i in range(self.n_layers):
                weights, inputs = _split(inputs, 2)
                ws.append(weights)
            bs = []
            for i in range(self.n_layers):
                biases, inputs = _split(inputs, 2)
                bs.append(biases)
            xs = inputs
            hy, ys = functions.n_step_rnn(
                self.n_layers, self.dropout, hx, ws, bs, xs,
                activation=self.activation)
            return (hy, ) + ys

        gradient_check.check_backward(
            f, args, grads, rtol=1e-2, atol=5e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.hx, self.xs, self.ws, self.bs,
                            self.dhy, self.dys)

    @condition.retry(3)
    def test_backward_partially_none_cpu(self):
        self.dys[1] = None
        self.check_backward(self.hx, self.xs, self.ws, self.bs,
                            None, self.dys)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.check_backward(
                _to_gpu(self.hx),
                _to_gpu(self.xs),
                _to_gpu(self.ws),
                _to_gpu(self.bs),
                _to_gpu(self.dhy),
                _to_gpu(self.dys))

    @attr.gpu
    @condition.retry(3)
    def test_backward_partially_none_gpu(self):
        self.dys[1] = None
        with chainer.using_config('use_cudnn', 'always'):
            self.check_backward(
                _to_gpu(self.hx),
                _to_gpu(self.xs),
                _to_gpu(self.ws),
                _to_gpu(self.bs),
                None,
                _to_gpu(self.dys))

    def call_forward(self, train):
        hx = _wrap_variable(_to_gpu(self.hx))
        xs = _wrap_variable(_to_gpu(self.xs))
        ws = _wrap_variable(_to_gpu(self.ws))
        bs = _wrap_variable(_to_gpu(self.bs))
        with chainer.using_config('enable_backprop', train), \
                chainer.using_config('train', train):
            return functions.n_step_rnn(
                self.n_layers, self.dropout, hx, ws, bs, xs)

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

    def check_call_cudnn_backward_training(self, use_cudnn):
        with chainer.using_config('use_cudnn', use_cudnn):
            expect = chainer.should_use_cudnn('>=auto', 5000)
            hy, ys = self.call_forward(True)
            hy.grad = _to_gpu(self.dhy)
            with testing.patch('cupy.cudnn.rnn_backward_weights') as func:
                hy.backward()
            assert func.called == expect

    @attr.cudnn
    def test_call_cudnn_backward_training(self):
        self.check_call_cudnn_backward_training('always')
        self.check_call_cudnn_backward_training('never')
        self.check_call_cudnn_backward_training('auto')

    def check_call_cudnn_backward_inference(self, use_cudnn):
        with chainer.using_config('use_cudnn', use_cudnn), \
                chainer.using_config('train', False):
            hx = _wrap_variable(_to_gpu(self.hx))
            xs = _wrap_variable(_to_gpu(self.xs))
            ws = _wrap_variable(_to_gpu(self.ws))
            bs = _wrap_variable(_to_gpu(self.bs))
            hy, ys = functions.n_step_rnn(
                self.n_layers, self.dropout, hx, ws, bs, xs)

            hy.grad = _to_gpu(self.dhy)
            if chainer.should_use_cudnn('>=auto', 5000):
                with self.assertRaises(RuntimeError):
                    hy.backward()
            else:
                with testing.patch('cupy.cudnn.rnn_backward_weights') as func:
                    hy.backward()
                assert not func.called

    @attr.cudnn
    def test_call_cudnn_backward_inference(self):
        # see chainer/5943
        self.check_call_cudnn_backward_inference('always')
        self.check_call_cudnn_backward_inference('never')
        self.check_call_cudnn_backward_inference('auto')

    def check_inconsistent_input_size(self, h_data, xs_data, ws_data, bs_data):
        h = _wrap_variable(h_data)
        xs = _wrap_variable(xs_data)
        ws = _wrap_variable(ws_data)
        bs = _wrap_variable(bs_data)
        with self.assertRaises(ValueError):
            functions.n_step_rnn(
                self.n_layers, self.dropout, h, ws, bs, xs,
                activation=self.activation)

    def test_inconsistent_input_size_cpu(self):
        x_in_size = 4  # inconsistent in_size with that of ws.
        x_shape = [(b, x_in_size) for b in self.batches]
        xs = _shaped_random(x_shape)
        self.check_inconsistent_input_size(self.hx, xs, self.ws, self.bs)

    def check_inconsistent_input_size_gpu(self, use_cudnn):
        x_in_size = 4  # inconsistent in_size with that of ws.
        x_shape = [(b, x_in_size) for b in self.batches]
        xs = _shaped_random(x_shape)

        hx = _to_gpu(self.hx)
        xs = _to_gpu(xs)
        ws = _to_gpu(self.ws)
        bs = _to_gpu(self.bs)
        with chainer.using_config('use_cudnn', use_cudnn):
            self.check_inconsistent_input_size(hx, xs, ws, bs)

    @attr.gpu
    def test_inconsistent_input_size_gpu_cudnn_always(self):
        self.check_inconsistent_input_size_gpu('always')

    @attr.gpu
    def test_inconsistent_input_size_gpu_cudnn_never(self):
        self.check_inconsistent_input_size_gpu('never')


@testing.parameterize(*testing.product({
    'activation': ['tanh', 'relu']
}))
class TestNStepBiRNN(unittest.TestCase):

    batches = [3, 2, 1]
    length = len(batches)
    in_size = 3
    out_size = 2
    n_layers = 2
    dropout = 0.0

    def setUp(self):
        self.xs = _shaped_random([(b, self.in_size) for b in self.batches])
        h_shape = (self.n_layers * 2, self.batches[0], self.out_size)
        self.hx = _shaped_random(h_shape)

        i = self.in_size
        o = self.out_size
        self.ws = []
        self.bs = []
        # First layer has the different shape
        for di in range(2):
            self.ws.append(_shaped_random([(o, i), (o, o)]))
            self.bs.append(_shaped_random([o, o]))
        # Rest layers
        for _ in range(self.n_layers - 1):
            for di in range(2):
                self.ws.append(_shaped_random([(o, o * 2), (o, o)]))
                self.bs.append(_shaped_random([o, o]))

        self.dys = _shaped_random(
            [(b, self.out_size * 2) for b in self.batches])
        self.dhy = _shaped_random(h_shape)

    def check_forward(
            self, h_data, xs_data, ws_data, bs_data):
        h = _wrap_variable(h_data)
        xs = _wrap_variable(xs_data)
        ws = _wrap_variable(ws_data)
        bs = _wrap_variable(bs_data)
        hy, ys = functions.n_step_birnn(
            self.n_layers, self.dropout, h, ws, bs, xs,
            activation=self.activation)

        xs_next = self.xs
        e_hy = self.hx.copy()
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
                if self.activation == 'tanh':
                    e_h = numpy.tanh(x.dot(w[0].T) +
                                     h_prev.dot(w[1].T) + b[0] + b[1])
                elif self.activation == 'relu':
                    e_h = _relu(x.dot(w[0].T) +
                                h_prev.dot(w[1].T) + b[0] + b[1])

                e_hy[layer_idx, :batch] = e_h
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
                if self.activation == 'tanh':
                    e_h = numpy.tanh(x.dot(w[0].T) +
                                     h_prev.dot(w[1].T) + b[0] + b[1])
                elif self.activation == 'relu':
                    e_h = _relu(x.dot(w[0].T) +
                                h_prev.dot(w[1].T) + b[0] + b[1])

                e_hy[layer_idx, :batch] = e_h
                xb.append(e_h)
            xb.reverse()
            xs_next = [numpy.concatenate([hfi, hbi], axis=1) for (hfi, hbi) in
                       zip(xf, xb)]

        for k, (ysi, xsi) in enumerate(zip(ys, xs_next)):
            testing.assert_allclose(ysi.data, xsi, rtol=1e-4, atol=1e-4)

        testing.assert_allclose(hy.data, e_hy, rtol=1e-4, atol=1e-4)

    def test_forward_cpu(self):
        self.check_forward(self.hx, self.xs, self.ws, self.bs)

    def check_forward_gpu(self, use_cudnn):
        with chainer.using_config('use_cudnn', use_cudnn):
            self.check_forward(
                _to_gpu(self.hx),
                _to_gpu(self.xs),
                _to_gpu(self.ws),
                _to_gpu(self.bs))

    @attr.gpu
    def test_forward_gpu_cudnn_always(self):
        self.check_forward_gpu('always')

    @attr.gpu
    def test_forward_gpu_cudnn_never(self):
        self.check_forward_gpu('never')

    def check_backward(self, h_data, xs_data, ws_data, bs_data,
                       dhy_data, dys_data):
        args = tuple([h_data, ] + sum(ws_data, []) + sum(bs_data, []) +
                     xs_data)
        grads = tuple([dhy_data, ] + dys_data)

        def f(*inputs):
            (hx, ), inputs = _split(inputs, 1)
            ws = []
            for i in range(self.n_layers * 2):
                weights, inputs = _split(inputs, 2)
                ws.append(weights)
            bs = []
            for i in range(self.n_layers * 2):
                biases, inputs = _split(inputs, 2)
                bs.append(biases)
            xs = inputs
            hy, ys = functions.n_step_birnn(
                self.n_layers, self.dropout, hx, ws, bs, xs,
                activation=self.activation)
            return (hy, ) + ys

        gradient_check.check_backward(
            f, args, grads, rtol=1e-2, atol=5e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.hx, self.xs, self.ws, self.bs,
                            self.dhy, self.dys)

    @condition.retry(3)
    def test_backward_partially_none_cpu(self):
        self.dys[1] = None
        self.check_backward(self.hx, self.xs, self.ws, self.bs,
                            None, self.dys)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.check_backward(
                _to_gpu(self.hx),
                _to_gpu(self.xs),
                _to_gpu(self.ws),
                _to_gpu(self.bs),
                _to_gpu(self.dhy),
                _to_gpu(self.dys))

    @attr.gpu
    @condition.retry(3)
    def test_backward_partially_none_gpu(self):
        self.dys[1] = None
        with chainer.using_config('use_cudnn', 'always'):
            self.check_backward(
                _to_gpu(self.hx),
                _to_gpu(self.xs),
                _to_gpu(self.ws),
                _to_gpu(self.bs),
                None,
                _to_gpu(self.dys))

    def call_forward(self, train):
        hx = _wrap_variable(_to_gpu(self.hx))
        xs = _wrap_variable(_to_gpu(self.xs))
        ws = _wrap_variable(_to_gpu(self.ws))
        bs = _wrap_variable(_to_gpu(self.bs))
        with chainer.using_config('enable_backprop', train), \
                chainer.using_config('train', train):
            return functions.n_step_birnn(
                self.n_layers, self.dropout, hx, ws, bs, xs)

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
            hy, ys = self.call_forward(True)
            hy.grad = _to_gpu(self.dhy)
            with testing.patch('cupy.cudnn.rnn_backward_weights') as func:
                hy.backward()
            assert func.called == expect

    @attr.cudnn
    def test_call_cudnn_backward(self):
        self.check_call_cudnn_backward('always')
        self.check_call_cudnn_backward('never')
        self.check_call_cudnn_backward('auto')

    def check_inconsistent_input_size(self, h_data, xs_data, ws_data, bs_data):
        h = _wrap_variable(h_data)
        xs = _wrap_variable(xs_data)
        ws = _wrap_variable(ws_data)
        bs = _wrap_variable(bs_data)
        with self.assertRaises(ValueError):
            functions.n_step_birnn(
                self.n_layers, self.dropout, h, ws, bs, xs,
                activation=self.activation)

    def test_inconsistent_input_size_cpu(self):
        x_in_size = 4  # inconsistent in_size with that of ws.
        x_shape = [(b, x_in_size) for b in self.batches]
        xs = _shaped_random(x_shape)
        self.check_inconsistent_input_size(self.hx, xs, self.ws, self.bs)

    def check_inconsistent_input_size_gpu(self, use_cudnn):
        x_in_size = 4  # inconsistent in_size with that of ws.
        x_shape = [(b, x_in_size) for b in self.batches]
        xs = _shaped_random(x_shape)

        hx = _to_gpu(self.hx)
        xs = _to_gpu(xs)
        ws = _to_gpu(self.ws)
        bs = _to_gpu(self.bs)
        with chainer.using_config('use_cudnn', use_cudnn):
            self.check_inconsistent_input_size(hx, xs, ws, bs)

    @attr.gpu
    def test_inconsistent_input_size_gpu_cudnn_always(self):
        self.check_inconsistent_input_size_gpu('always')

    @attr.gpu
    def test_inconsistent_input_size_gpu_cudnn_never(self):
        self.check_inconsistent_input_size_gpu('never')


testing.run_module(__name__, __file__)
