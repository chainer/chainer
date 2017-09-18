import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr


def sigmoid(x):
    return numpy.tanh(x * 0.5) * 0.5 + 0.5


@testing.parameterize(*testing.product({
    'hidden_none': [True, False],
}))
class TestNStepLSTM(unittest.TestCase):

    lengths = [3, 1, 2]
    n_layer = 2
    in_size = 3
    out_size = 2
    dropout = 0.0

    def setUp(self):
        shape = (self.n_layer, len(self.lengths), self.out_size)
        if self.hidden_none:
            self.h = self.c = numpy.zeros(shape, 'f')
        else:
            self.h = numpy.random.uniform(-1, 1, shape).astype('f')
            self.c = numpy.random.uniform(-1, 1, shape).astype('f')
        self.xs = [
            numpy.random.uniform(-1, 1, (l, self.in_size)).astype('f')
            for l in self.lengths]

        self.gh = numpy.random.uniform(-1, 1, shape).astype('f')
        self.gc = numpy.random.uniform(-1, 1, shape).astype('f')
        self.gys = [
            numpy.random.uniform(-1, 1, (l, self.out_size)).astype('f')
            for l in self.lengths]
        self.rnn = links.NStepLSTM(
            self.n_layer, self.in_size, self.out_size, self.dropout)

        for layer in self.rnn:
            for p in layer.params():
                p.data[...] = numpy.random.uniform(-1, 1, p.data.shape)
        self.rnn.cleargrads()

    def check_forward(self, h_data, c_data, xs_data):
        if self.hidden_none:
            h = c = None
        else:
            h = chainer.Variable(h_data)
            c = chainer.Variable(c_data)
        xs = [chainer.Variable(x) for x in xs_data]
        hy, cy, ys = self.rnn(h, c, xs)

        self.assertEqual(hy.data.shape, h_data.shape)
        self.assertEqual(cy.data.shape, c_data.shape)
        self.assertEqual(len(xs), len(ys))
        for x, y in zip(xs, ys):
            self.assertEqual(len(x.data), len(y.data))
            self.assertEqual(y.data.shape[1], self.out_size)

        self.rnn.to_cpu()

        for batch, seq in enumerate(self.xs):
            for layer in range(self.n_layer):
                p = self.rnn[layer]
                h_prev = self.h[layer, batch]
                c_prev = self.c[layer, batch]
                hs = []
                for x in seq:
                    i = sigmoid(x.dot(p.w0.data.T) + h_prev.dot(p.w4.data.T) +
                                p.b0.data + p.b4.data)
                    f = sigmoid(x.dot(p.w1.data.T) + h_prev.dot(p.w5.data.T) +
                                p.b1.data + p.b5.data)
                    c_bar = numpy.tanh(
                        x.dot(p.w2.data.T) + h_prev.dot(p.w6.data.T) +
                        p.b2.data + p.b6.data)
                    o = sigmoid(x.dot(p.w3.data.T) + h_prev.dot(p.w7.data.T) +
                                p.b3.data + p.b7.data)
                    e_c = (f * c_prev + i * c_bar)
                    e_h = o * numpy.tanh(e_c)

                    h_prev = e_h
                    c_prev = e_c
                    hs.append(e_h)

                seq = hs
                testing.assert_allclose(hy.data[layer, batch], h_prev)
                testing.assert_allclose(cy.data[layer, batch], c_prev)

            for y, ey in zip(ys[batch].data, seq):
                testing.assert_allclose(y, ey)

    def test_forward_cpu_train(self):
        with chainer.using_config('train', True):
            self.check_forward(self.h, self.c, self.xs)

    @attr.gpu
    def test_forward_gpu_train(self):
        self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'always'), \
                chainer.using_config('train', True):
            self.check_forward(
                cuda.to_gpu(self.h),
                cuda.to_gpu(self.c),
                [cuda.to_gpu(x) for x in self.xs])

    def test_forward_cpu_test(self):
        with chainer.using_config('train', False):
            self.check_forward(self.h, self.c, self.xs)

    @attr.gpu
    def test_forward_gpu_test(self):
        self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'always'), \
                chainer.using_config('train', False):
            self.check_forward(
                cuda.to_gpu(self.h),
                cuda.to_gpu(self.c),
                [cuda.to_gpu(x) for x in self.xs])

    def check_backward(
            self, h_data, c_data, xs_data, gh_data, gc_data, gys_data):

        def fun(*args):
            if self.hidden_none:
                h = c = None
                xs = args
            else:
                h, c = args[:2]
                xs = args[2:]
            hy, cy, ys = self.rnn(h, c, xs)
            return tuple([hy, cy] + list(ys))

        params = []
        for layer in self.rnn:
            for p in layer.params():
                params.append(p)

        if self.hidden_none:
            in_data = xs_data
        else:
            in_data = [h_data, c_data] + xs_data
            gradient_check.check_backward(
                fun, tuple(in_data),
                tuple([gh_data, gc_data] + gys_data),
                tuple(params), eps=1e-2, rtol=1e-3, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(
            self.h, self.c, self.xs, self.gh, self.gc, self.gys)

    @attr.gpu
    def test_backward_gpu(self):
        self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'always'):
            self.check_backward(
                cuda.to_gpu(self.h),
                cuda.to_gpu(self.c),
                [cuda.to_gpu(x) for x in self.xs],
                cuda.to_gpu(self.gh),
                cuda.to_gpu(self.gc),
                [cuda.to_gpu(gy) for gy in self.gys])


@testing.parameterize(*testing.product({
    'hidden_none': [True, False],
}))
class TestNStepBiLSTM(unittest.TestCase):

    lengths = [3, 1, 2]
    n_layer = 2
    in_size = 3
    out_size = 2
    dropout = 0.0

    def setUp(self):
        shape = (self.n_layer * 2, len(self.lengths), self.out_size)
        if self.hidden_none:
            self.h = self.c = numpy.zeros(shape, 'f')
        else:
            self.h = numpy.random.uniform(-1, 1, shape).astype('f')
            self.c = numpy.random.uniform(-1, 1, shape).astype('f')
        self.xs = [
            numpy.random.uniform(-1, 1, (l, self.in_size)).astype('f')
            for l in self.lengths]

        self.gh = numpy.random.uniform(-1, 1, shape).astype('f')
        self.gc = numpy.random.uniform(-1, 1, shape).astype('f')
        self.gys = [
            numpy.random.uniform(-1, 1, (l, self.out_size * 2)).astype('f')
            for l in self.lengths]
        self.rnn = links.NStepBiLSTM(
            self.n_layer, self.in_size, self.out_size, self.dropout)

        for layer in self.rnn:
            for p in layer.params():
                p.data[...] = numpy.random.uniform(-1, 1, p.data.shape)
        self.rnn.cleargrads()

    def check_forward(self, h_data, c_data, xs_data):
        if self.hidden_none:
            h = c = None
        else:
            h = chainer.Variable(h_data)
            c = chainer.Variable(c_data)
        xs = [chainer.Variable(x) for x in xs_data]
        hy, cy, ys = self.rnn(h, c, xs)

        self.assertEqual(hy.data.shape, h_data.shape)
        self.assertEqual(cy.data.shape, c_data.shape)
        self.assertEqual(len(xs), len(ys))
        for x, y in zip(xs, ys):
            self.assertEqual(len(x.data), len(y.data))
            self.assertEqual(y.data.shape[1], self.out_size * 2)

        self.rnn.to_cpu()

        for batch, seq in enumerate(self.xs):
            for layer in range(self.n_layer):
                # forward
                di = 0
                layer_idx = layer * 2 + di
                p = self.rnn[layer_idx]
                h_prev = self.h[layer_idx, batch]
                c_prev = self.c[layer_idx, batch]
                hs_f = []
                for x in seq:
                    i = sigmoid(x.dot(p.w0.data.T) +
                                h_prev.dot(p.w4.data.T) +
                                p.b0.data + p.b4.data)
                    f = sigmoid(x.dot(p.w1.data.T) +
                                h_prev.dot(p.w5.data.T) +
                                p.b1.data + p.b5.data)
                    c_bar = numpy.tanh(x.dot(p.w2.data.T) +
                                       h_prev.dot(p.w6.data.T) +
                                       p.b2.data + p.b6.data)
                    o = sigmoid(x.dot(p.w3.data.T) + h_prev.dot(p.w7.data.T) +
                                p.b3.data + p.b7.data)
                    e_c = (f * c_prev + i * c_bar)
                    e_h = o * numpy.tanh(e_c)

                    h_prev = e_h
                    c_prev = e_c
                    hs_f.append(e_h)

                testing.assert_allclose(hy.data[layer_idx, batch], h_prev)
                testing.assert_allclose(cy.data[layer_idx, batch], c_prev)

                # backward
                di = 1
                layer_idx = layer * 2 + di
                p = self.rnn[layer_idx]
                h_prev = self.h[layer_idx, batch]
                c_prev = self.c[layer_idx, batch]
                hs_b = []
                for x in reversed(seq):
                    i = sigmoid(x.dot(p.w0.data.T) +
                                h_prev.dot(p.w4.data.T) +
                                p.b0.data + p.b4.data)
                    f = sigmoid(x.dot(p.w1.data.T) +
                                h_prev.dot(p.w5.data.T) +
                                p.b1.data + p.b5.data)
                    c_bar = numpy.tanh(x.dot(p.w2.data.T) +
                                       h_prev.dot(p.w6.data.T) +
                                       p.b2.data + p.b6.data)
                    o = sigmoid(x.dot(p.w3.data.T) + h_prev.dot(p.w7.data.T) +
                                p.b3.data + p.b7.data)
                    e_c = (f * c_prev + i * c_bar)
                    e_h = o * numpy.tanh(e_c)

                    h_prev = e_h
                    c_prev = e_c
                    hs_b.append(e_h)

                testing.assert_allclose(hy.data[layer_idx, batch], h_prev)
                testing.assert_allclose(cy.data[layer_idx, batch], c_prev)

                hs_b.reverse()
                seq = [numpy.concatenate([hfi, hbi], axis=0) for (hfi, hbi)
                       in zip(hs_f, hs_b)]

            for y, ey in zip(ys[batch].data, seq):
                testing.assert_allclose(y, ey)

    def test_forward_cpu_train(self):
        with chainer.using_config('train', True):
            self.check_forward(self.h, self.c, self.xs)

    @attr.gpu
    def test_forward_gpu_train(self):
        self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'always'), \
                chainer.using_config('train', True):
            self.check_forward(
                cuda.to_gpu(self.h),
                cuda.to_gpu(self.c),
                [cuda.to_gpu(x) for x in self.xs])

    def test_forward_cpu_test(self):
        with chainer.using_config('train', False):
            self.check_forward(self.h, self.c, self.xs)

    @attr.gpu
    def test_forward_gpu_test(self):
        self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'always'), \
                chainer.using_config('train', False):
            self.check_forward(
                cuda.to_gpu(self.h),
                cuda.to_gpu(self.c),
                [cuda.to_gpu(x) for x in self.xs])

    def check_backward(
            self, h_data, c_data, xs_data, gh_data, gc_data, gys_data):

        def fun(*args):
            if self.hidden_none:
                h = c = None
                xs = args
            else:
                h, c = args[:2]
                xs = args[2:]
            hy, cy, ys = self.rnn(h, c, xs)
            return tuple([hy, cy] + list(ys))

        params = []
        for layer in self.rnn:
            for p in layer.params():
                params.append(p)

        if self.hidden_none:
            in_data = xs_data
        else:
            in_data = [h_data, c_data] + xs_data
        gradient_check.check_backward(
            fun, tuple(in_data),
            tuple([gh_data, gc_data] + gys_data),
            tuple(params), eps=1e-2, rtol=1e-3, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(
            self.h, self.c, self.xs, self.gh, self.gc, self.gys)

    @attr.gpu
    def test_backward_gpu(self):
        self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'auto'):
            self.check_backward(
                cuda.to_gpu(self.h),
                cuda.to_gpu(self.c),
                [cuda.to_gpu(x) for x in self.xs],
                cuda.to_gpu(self.gh),
                cuda.to_gpu(self.gc),
                [cuda.to_gpu(gy) for gy in self.gys])


testing.run_module(__name__, __file__)
