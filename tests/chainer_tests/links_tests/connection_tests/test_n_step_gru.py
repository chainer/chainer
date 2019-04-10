import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr


def sigmoid(x):
    return numpy.tanh(x * 0.5) * 0.5 + 0.5


@testing.parameterize(*testing.product({
    'hidden_none': [True, False],
}))
class TestNStepGRU(unittest.TestCase):

    lengths = [3, 1, 2]
    n_layer = 2
    in_size = 3
    out_size = 2
    dropout = 0.0

    def setUp(self):
        shape = (self.n_layer, len(self.lengths), self.out_size)
        if self.hidden_none:
            self.h = numpy.zeros(shape, 'f')
        else:
            self.h = numpy.random.uniform(-1, 1, shape).astype('f')
        self.xs = [
            numpy.random.uniform(-1, 1, (l, self.in_size)).astype('f')
            for l in self.lengths]

        self.gh = numpy.random.uniform(-1, 1, shape).astype('f')
        self.gys = [
            numpy.random.uniform(-1, 1, (l, self.out_size)).astype('f')
            for l in self.lengths]
        self.rnn = links.NStepGRU(
            self.n_layer, self.in_size, self.out_size, self.dropout)

        for layer in self.rnn:
            for p in layer.params():
                p.array[...] = numpy.random.uniform(-1, 1, p.shape)
        self.rnn.cleargrads()

    def check_forward(self, h_data, xs_data):
        if self.hidden_none:
            h = None
        else:
            h = chainer.Variable(h_data)
        xs = [chainer.Variable(x) for x in xs_data]
        hy, ys = self.rnn(h, xs)

        assert hy.shape == h_data.shape
        assert len(xs) == len(ys)
        for x, y in zip(xs, ys):
            assert len(x) == len(y)
            assert y.shape[1] == self.out_size

        self.rnn.to_cpu()

        for batch, seq in enumerate(self.xs):
            for layer in range(self.n_layer):
                p = self.rnn[layer]
                h_prev = self.h[layer, batch]
                hs = []
                for x in seq:
                    # GRU
                    z = sigmoid(
                        x.dot(p.w1.array.T) + h_prev.dot(p.w4.array.T) +
                        p.b1.array + p.b4.array)
                    r = sigmoid(
                        x.dot(p.w0.array.T) + h_prev.dot(p.w3.array.T) +
                        p.b0.array + p.b3.array)
                    h_bar = numpy.tanh(
                        x.dot(p.w2.array.T) +
                        r * ((h_prev).dot(p.w5.array.T) + p.b5.array) +
                        p.b2.array)
                    e_h = (1 - z) * h_bar + z * h_prev

                    h_prev = e_h
                    hs.append(e_h)

                seq = hs
                testing.assert_allclose(hy.array[layer, batch], h_prev)

            for y, ey in zip(ys[batch].array, seq):
                testing.assert_allclose(y, ey)

    def test_forward_cpu_train(self):
        with chainer.using_config('train', True):
            self.check_forward(self.h, self.xs)

    @attr.gpu
    def test_forward_gpu_train(self):
        self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'always'), \
                chainer.using_config('train', True):
            self.check_forward(
                cuda.to_gpu(self.h),
                [cuda.to_gpu(x) for x in self.xs])

    def test_forward_cpu_test(self):
        with chainer.using_config('train', False):
            self.check_forward(self.h, self.xs)

    @attr.gpu
    def test_forward_gpu_test(self):
        self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'always'), \
                chainer.using_config('train', False):
            self.check_forward(
                cuda.to_gpu(self.h),
                [cuda.to_gpu(x) for x in self.xs])

    def check_multi_gpu_forward(self, train=True):
        # See chainer/chainer#6262
        # NStepGRU w/ cudnn & dropout should work on not current device
        msg = None
        rnn = self.rnn.copy('copy')
        rnn.dropout = .5
        with cuda.get_device_from_id(1):
            if self.hidden_none:
                h = None
            else:
                h = cuda.to_gpu(self.h)
            xs = [cuda.to_gpu(x) for x in self.xs]
            rnn = rnn.to_gpu()
        with cuda.get_device_from_id(0),\
                chainer.using_config('train', train),\
                chainer.using_config('use_cudnn', 'always'):
            try:
                rnn(h, xs)
            except Exception as e:
                msg = e
        assert msg is None

    @attr.cudnn
    @attr.multi_gpu(2)
    def test_multi_gpu_forward_training(self):
        self.check_multi_gpu_forward(True)

    @attr.cudnn
    @attr.multi_gpu(2)
    def test_multi_gpu_forward_test(self):
        self.check_multi_gpu_forward(False)

    def check_backward(
            self, h_data, xs_data, gh_data, gys_data):

        def fun(*args):
            if self.hidden_none:
                h = None
                xs = args
            else:
                h, = args[:1]
                xs = args[1:]
            hy, ys = self.rnn(h, xs)
            return tuple([hy, ] + list(ys))

        params = []
        for layer in self.rnn:
            for p in layer.params():
                params.append(p)

        if self.hidden_none:
            in_data = xs_data
        else:
            in_data = [h_data] + xs_data
        gradient_check.check_backward(
            fun, tuple(in_data),
            tuple([gh_data] + gys_data),
            tuple(params), eps=1e-2, rtol=1e-3, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.h, self.xs, self.gh, self.gys)

    @attr.gpu
    def test_backward_gpu(self):
        self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'auto'):
            self.check_backward(
                cuda.to_gpu(self.h),
                [cuda.to_gpu(x) for x in self.xs],
                cuda.to_gpu(self.gh),
                [cuda.to_gpu(gy) for gy in self.gys])

    def test_n_cells(self):
        assert self.rnn.n_cells == 1


@testing.parameterize(*testing.product({
    'hidden_none': [True, False],
}))
class TestNStepBiGRU(unittest.TestCase):

    lengths = [3, 1, 2]
    n_layer = 2
    in_size = 3
    out_size = 2
    dropout = 0.0

    def setUp(self):
        shape = (self.n_layer * 2, len(self.lengths), self.out_size)
        if self.hidden_none:
            self.h = numpy.zeros(shape, 'f')
        else:
            self.h = numpy.random.uniform(-1, 1, shape).astype('f')
        self.xs = [
            numpy.random.uniform(-1, 1, (l, self.in_size)).astype('f')
            for l in self.lengths]

        self.gh = numpy.random.uniform(-1, 1, shape).astype('f')
        self.gys = [
            numpy.random.uniform(-1, 1, (l, self.out_size * 2)).astype('f')
            for l in self.lengths]
        self.rnn = links.NStepBiGRU(
            self.n_layer, self.in_size, self.out_size, self.dropout)

        for layer in self.rnn:
            for p in layer.params():
                p.array[...] = numpy.random.uniform(-1, 1, p.shape)
        self.rnn.cleargrads()

    def check_forward(self, h_data, xs_data):
        if self.hidden_none:
            h = None
        else:
            h = chainer.Variable(h_data)
        xs = [chainer.Variable(x) for x in xs_data]
        hy, ys = self.rnn(h, xs)

        assert hy.shape == h_data.shape
        assert len(xs) == len(ys)
        for x, y in zip(xs, ys):
            assert len(x) == len(y)
            assert y.shape[1] == self.out_size * 2

        self.rnn.to_cpu()

        for batch, seq in enumerate(self.xs):
            for layer in range(self.n_layer):
                # forward
                di = 0
                layer_idx = layer * 2 + di
                p = self.rnn[layer_idx]
                h_prev = self.h[layer_idx, batch]
                hs_f = []
                for x in seq:
                    # GRU
                    z = sigmoid(
                        x.dot(p.w1.array.T) + h_prev.dot(p.w4.array.T) +
                        p.b1.array + p.b4.array)
                    r = sigmoid(
                        x.dot(p.w0.array.T) + h_prev.dot(p.w3.array.T) +
                        p.b0.array + p.b3.array)
                    h_bar = numpy.tanh(x.dot(p.w2.array.T) +
                                       r * ((h_prev).dot(p.w5.array.T) +
                                            p.b5.array) + p.b2.array)
                    e_h = (1 - z) * h_bar + z * h_prev

                    h_prev = e_h
                    hs_f.append(e_h)

                testing.assert_allclose(hy.array[layer_idx, batch], h_prev)

                # backward
                di = 1
                layer_idx = layer * 2 + di
                p = self.rnn[layer_idx]
                h_prev = self.h[layer_idx, batch]
                hs_b = []
                for x in reversed(seq):
                    # GRU
                    z = sigmoid(
                        x.dot(p.w1.array.T) + h_prev.dot(p.w4.array.T) +
                        p.b1.array + p.b4.array)
                    r = sigmoid(
                        x.dot(p.w0.array.T) + h_prev.dot(p.w3.array.T) +
                        p.b0.array + p.b3.array)
                    h_bar = numpy.tanh(x.dot(p.w2.array.T) +
                                       r * ((h_prev).dot(p.w5.array.T) +
                                            p.b5.array) + p.b2.array)
                    e_h = (1 - z) * h_bar + z * h_prev
                    h_prev = e_h
                    hs_b.append(e_h)
                testing.assert_allclose(hy.array[layer_idx, batch], h_prev)

                hs_b.reverse()
                seq = [numpy.concatenate([hfi, hbi], axis=0) for (hfi, hbi)
                       in zip(hs_f, hs_b)]

            for y, ey in zip(ys[batch].array, seq):
                testing.assert_allclose(y, ey)

    def test_forward_cpu_train(self):
        with chainer.using_config('train', True):
            self.check_forward(self.h, self.xs)

    @attr.gpu
    def test_forward_gpu_train(self):
        self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'always'), \
                chainer.using_config('train', True):
            self.check_forward(
                cuda.to_gpu(self.h),
                [cuda.to_gpu(x) for x in self.xs])

    def test_forward_cpu_test(self):
        with chainer.using_config('train', False):
            self.check_forward(self.h, self.xs)

    @attr.gpu
    def test_forward_gpu_test(self):
        self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'always'), \
                chainer.using_config('train', False):
            self.check_forward(
                cuda.to_gpu(self.h),
                [cuda.to_gpu(x) for x in self.xs])

    def check_multi_gpu_forward(self, train=True):
        # See chainer/chainer#6262
        # NStepBiGRU w/ cudnn and dropout should work on not current device
        msg = None
        rnn = self.rnn.copy('copy')
        rnn.dropout = .5
        with cuda.get_device_from_id(1):
            if self.hidden_none:
                h = None
            else:
                h = cuda.to_gpu(self.h)
            xs = [cuda.to_gpu(x) for x in self.xs]
            rnn = rnn.to_gpu()
        with cuda.get_device_from_id(0),\
                chainer.using_config('train', train),\
                chainer.using_config('use_cudnn', 'always'):
            try:
                rnn(h, xs)
            except Exception as e:
                msg = e
        assert msg is None

    @attr.cudnn
    @attr.multi_gpu(2)
    def test_multi_gpu_forward_training(self):
        self.check_multi_gpu_forward(True)

    @attr.cudnn
    @attr.multi_gpu(2)
    def test_multi_gpu_forward_test(self):
        self.check_multi_gpu_forward(False)

    def check_backward(
            self, h_data, xs_data, gh_data, gys_data):

        def fun(*args):
            if self.hidden_none:
                h = None
                xs = args
            else:
                h, = args[:1]
                xs = args[1:]
            hy, ys = self.rnn(h, xs)
            return tuple([hy, ] + list(ys))

        params = []
        for layer in self.rnn:
            for p in layer.params():
                params.append(p)

        if self.hidden_none:
            in_data = xs_data
        else:
            in_data = [h_data] + xs_data
        gradient_check.check_backward(
            fun, tuple(in_data),
            tuple([gh_data] + gys_data),
            tuple(params), eps=1e-2, rtol=1e-3, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(
            self.h, self.xs, self.gh, self.gys)

    @attr.gpu
    def test_backward_gpu(self):
        self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'auto'):
            self.check_backward(
                cuda.to_gpu(self.h),
                [cuda.to_gpu(x) for x in self.xs],
                cuda.to_gpu(self.gh),
                [cuda.to_gpu(gy) for gy in self.gys])

    def test_n_cells(self):
        assert self.rnn.n_cells == 1


testing.run_module(__name__, __file__)
