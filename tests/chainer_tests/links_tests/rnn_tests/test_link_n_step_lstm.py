import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import initializers
from chainer import testing
from chainer.testing import attr


def sigmoid(x):
    return numpy.tanh(x * 0.5) * 0.5 + 0.5


@testing.parameterize(*testing.product({
    'hidden_none': [True, False],
}))
class TestNStepLSTM(unittest.TestCase):

    lengths = [3, 1, 2]
    n_layers = 2
    in_size = 3
    out_size = 2
    dropout = 0.0

    def setUp(self):
        shape = (self.n_layers, len(self.lengths), self.out_size)
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
            self.n_layers, self.in_size, self.out_size, self.dropout)

        for layer in self.rnn:
            for p in layer.params():
                p.array[...] = numpy.random.uniform(-1, 1, p.shape)
        self.rnn.cleargrads()

    def check_forward(self, h_data, c_data, xs_data):
        if self.hidden_none:
            h = c = None
        else:
            h = chainer.Variable(h_data)
            c = chainer.Variable(c_data)
        xs = [chainer.Variable(x) for x in xs_data]
        hy, cy, ys = self.rnn(h, c, xs)

        assert hy.shape == h_data.shape
        assert cy.shape == c_data.shape
        assert len(xs) == len(ys)
        for x, y in zip(xs, ys):
            assert len(x) == len(y)
            assert y.shape[1] == self.out_size

        with testing.assert_warns(DeprecationWarning):
            self.rnn.to_cpu()

        for batch, seq in enumerate(self.xs):
            for layer in range(self.n_layers):
                p = self.rnn[layer]
                h_prev = self.h[layer, batch]
                c_prev = self.c[layer, batch]
                hs = []
                for x in seq:
                    i = sigmoid(
                        x.dot(p.w0.array.T) + h_prev.dot(p.w4.array.T) +
                        p.b0.array + p.b4.array)
                    f = sigmoid(
                        x.dot(p.w1.array.T) + h_prev.dot(p.w5.array.T) +
                        p.b1.array + p.b5.array)
                    c_bar = numpy.tanh(
                        x.dot(p.w2.array.T) + h_prev.dot(p.w6.array.T) +
                        p.b2.array + p.b6.array)
                    o = sigmoid(
                        x.dot(p.w3.array.T) + h_prev.dot(p.w7.array.T) +
                        p.b3.array + p.b7.array)
                    e_c = (f * c_prev + i * c_bar)
                    e_h = o * numpy.tanh(e_c)

                    h_prev = e_h
                    c_prev = e_c
                    hs.append(e_h)

                seq = hs
                testing.assert_allclose(hy.array[layer, batch], h_prev)
                testing.assert_allclose(cy.array[layer, batch], c_prev)

            for y, ey in zip(ys[batch].array, seq):
                testing.assert_allclose(y, ey)

    def test_forward_cpu_train(self):
        with chainer.using_config('train', True):
            self.check_forward(self.h, self.c, self.xs)

    @attr.gpu
    def test_forward_gpu_train(self):
        with testing.assert_warns(DeprecationWarning):
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
        with testing.assert_warns(DeprecationWarning):
            self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'always'), \
                chainer.using_config('train', False):
            self.check_forward(
                cuda.to_gpu(self.h),
                cuda.to_gpu(self.c),
                [cuda.to_gpu(x) for x in self.xs])

    @attr.multi_gpu(2)
    def test_forward_nonzero_gpu_test(self):
        # Issue #5347
        # to_gpu should work without setting the current device
        with testing.assert_warns(DeprecationWarning):
            self.rnn.to_gpu(1)
        with chainer.using_config('use_cudnn', 'always'), \
                chainer.using_config('train', False):
            self.check_forward(
                cuda.to_gpu(self.h, 1),
                cuda.to_gpu(self.c, 1),
                [cuda.to_gpu(x, 1) for x in self.xs])

    def check_multi_gpu_forward(self, train=True):
        # See chainer/chainer#6262
        # NStepLSTM w/ cudnn & dropout should work on not current device
        msg = None
        rnn = self.rnn.copy('copy')
        rnn.dropout = .5
        with cuda.get_device_from_id(1):
            if self.hidden_none:
                h = None
            else:
                h = cuda.to_gpu(self.h)
            c = cuda.to_gpu(self.c)
            xs = [cuda.to_gpu(x) for x in self.xs]
            with testing.assert_warns(DeprecationWarning):
                rnn = rnn.to_gpu()
        with cuda.get_device_from_id(0),\
                chainer.using_config('train', train),\
                chainer.using_config('use_cudnn', 'always'):
            try:
                rnn(h, c, xs)
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
        with testing.assert_warns(DeprecationWarning):
            self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'always'):
            self.check_backward(
                cuda.to_gpu(self.h),
                cuda.to_gpu(self.c),
                [cuda.to_gpu(x) for x in self.xs],
                cuda.to_gpu(self.gh),
                cuda.to_gpu(self.gc),
                [cuda.to_gpu(gy) for gy in self.gys])

    def test_n_cells(self):
        self.assertEqual(self.rnn.n_cells, 2)
        assert self.rnn.n_cells == 2


@testing.parameterize(*testing.product({
    'hidden_none': [True, False],
}))
class TestNStepBiLSTM(unittest.TestCase):

    lengths = [3, 1, 2]
    n_layers = 2
    in_size = 3
    out_size = 2
    dropout = 0.0

    def setUp(self):
        shape = (self.n_layers * 2, len(self.lengths), self.out_size)
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
            self.n_layers, self.in_size, self.out_size, self.dropout)

        for layer in self.rnn:
            for p in layer.params():
                p.array[...] = numpy.random.uniform(-1, 1, p.shape)
        self.rnn.cleargrads()

    def check_forward(self, h_data, c_data, xs_data):
        if self.hidden_none:
            h = c = None
        else:
            h = chainer.Variable(h_data)
            c = chainer.Variable(c_data)
        xs = [chainer.Variable(x) for x in xs_data]
        hy, cy, ys = self.rnn(h, c, xs)

        assert hy.shape == h_data.shape
        assert cy.shape == c_data.shape
        assert len(xs) == len(ys)
        for x, y in zip(xs, ys):
            assert len(x) == len(y)
            assert y.shape[1] == self.out_size * 2

        with testing.assert_warns(DeprecationWarning):
            self.rnn.to_cpu()

        for batch, seq in enumerate(self.xs):
            for layer in range(self.n_layers):
                # forward
                di = 0
                layer_idx = layer * 2 + di
                p = self.rnn[layer_idx]
                h_prev = self.h[layer_idx, batch]
                c_prev = self.c[layer_idx, batch]
                hs_f = []
                for x in seq:
                    i = sigmoid(x.dot(p.w0.array.T) +
                                h_prev.dot(p.w4.array.T) +
                                p.b0.array + p.b4.array)
                    f = sigmoid(x.dot(p.w1.array.T) +
                                h_prev.dot(p.w5.array.T) +
                                p.b1.array + p.b5.array)
                    c_bar = numpy.tanh(x.dot(p.w2.array.T) +
                                       h_prev.dot(p.w6.array.T) +
                                       p.b2.array + p.b6.array)
                    o = sigmoid(
                        x.dot(p.w3.array.T) + h_prev.dot(p.w7.array.T) +
                        p.b3.array + p.b7.array)
                    e_c = (f * c_prev + i * c_bar)
                    e_h = o * numpy.tanh(e_c)

                    h_prev = e_h
                    c_prev = e_c
                    hs_f.append(e_h)

                testing.assert_allclose(hy.array[layer_idx, batch], h_prev)
                testing.assert_allclose(cy.array[layer_idx, batch], c_prev)

                # backward
                di = 1
                layer_idx = layer * 2 + di
                p = self.rnn[layer_idx]
                h_prev = self.h[layer_idx, batch]
                c_prev = self.c[layer_idx, batch]
                hs_b = []
                for x in reversed(seq):
                    i = sigmoid(x.dot(p.w0.array.T) +
                                h_prev.dot(p.w4.array.T) +
                                p.b0.array + p.b4.array)
                    f = sigmoid(x.dot(p.w1.array.T) +
                                h_prev.dot(p.w5.array.T) +
                                p.b1.array + p.b5.array)
                    c_bar = numpy.tanh(x.dot(p.w2.array.T) +
                                       h_prev.dot(p.w6.array.T) +
                                       p.b2.array + p.b6.array)
                    o = sigmoid(
                        x.dot(p.w3.array.T) + h_prev.dot(p.w7.array.T) +
                        p.b3.array + p.b7.array)
                    e_c = (f * c_prev + i * c_bar)
                    e_h = o * numpy.tanh(e_c)

                    h_prev = e_h
                    c_prev = e_c
                    hs_b.append(e_h)

                testing.assert_allclose(hy.array[layer_idx, batch], h_prev)
                testing.assert_allclose(cy.array[layer_idx, batch], c_prev)

                hs_b.reverse()
                seq = [numpy.concatenate([hfi, hbi], axis=0) for (hfi, hbi)
                       in zip(hs_f, hs_b)]

            for y, ey in zip(ys[batch].array, seq):
                testing.assert_allclose(y, ey)

    def test_forward_cpu_train(self):
        with chainer.using_config('train', True):
            self.check_forward(self.h, self.c, self.xs)

    @attr.gpu
    def test_forward_gpu_train(self):
        with testing.assert_warns(DeprecationWarning):
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
        with testing.assert_warns(DeprecationWarning):
            self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'always'), \
                chainer.using_config('train', False):
            self.check_forward(
                cuda.to_gpu(self.h),
                cuda.to_gpu(self.c),
                [cuda.to_gpu(x) for x in self.xs])

    def check_multi_gpu_forward(self, train=True):
        # See chainer/chainer#6262
        # NStepBiLSTM w/ cudnn & dropout should work on not current device
        msg = None
        rnn = self.rnn.copy('copy')
        rnn.dropout = .5
        with cuda.get_device_from_id(1):
            if self.hidden_none:
                h = None
            else:
                h = cuda.to_gpu(self.h)
            c = cuda.to_gpu(self.c)
            xs = [cuda.to_gpu(x) for x in self.xs]
            with testing.assert_warns(DeprecationWarning):
                rnn = rnn.to_gpu()
        with cuda.get_device_from_id(0),\
                chainer.using_config('train', train),\
                chainer.using_config('use_cudnn', 'always'):
            try:
                rnn(h, c, xs)
            except Exception as e:
                msg = e
        assert msg is None

    @attr.gpu
    @attr.multi_gpu(2)
    def test_multi_gpu_forward_training(self):
        self.check_multi_gpu_forward(True)

    @attr.gpu
    @attr.multi_gpu(2)
    def test_multi_gpu_forward_test(self):
        self.check_multi_gpu_forward(False)

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
        with testing.assert_warns(DeprecationWarning):
            self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'auto'):
            self.check_backward(
                cuda.to_gpu(self.h),
                cuda.to_gpu(self.c),
                [cuda.to_gpu(x) for x in self.xs],
                cuda.to_gpu(self.gh),
                cuda.to_gpu(self.gc),
                [cuda.to_gpu(gy) for gy in self.gys])

    def test_n_cells(self):
        assert self.rnn.n_cells == 2


@testing.parameterize(
    *testing.product(
        {
            'dtype': [numpy.float32, numpy.float64],
            'initializer': ['random'],
            'use_bi_direction': [True, False]
        }
    )
)
class TestInitialization(unittest.TestCase):
    def setUp(self):
        if self.initializer is None:
            initializer = initializers.constant.Zero()

        elif self.initializer == 'random':
            initializer = initializers.GlorotUniform()

        self.lateral_init = numpy.zeros((10, 10), dtype=self.dtype)
        self.upward_init = numpy.zeros((10, 10), dtype=self.dtype)
        self.bias_init = numpy.zeros((10, 1), dtype=self.dtype)
        self.forget_bias_init = numpy.zeros((10, 1), dtype=self.dtype)

        initializer(self.lateral_init)
        initializer(self.upward_init)
        initializer(self.bias_init)
        initializer(self.forget_bias_init)
        print('#lateral: ', self.lateral_init)
        print('#upward: ', self.upward_init)

        # FIXME (himkt) .reshape(-1) is a workaronud
        self.bias_init = self.bias_init.reshape(-1)
        self.forget_bias_init = self.forget_bias_init.reshape(-1)

        with chainer.using_config('dtype', self.dtype):
            if self.use_bi_direction:
                link = links.NStepBiLSTM
            else:
                link = links.NStepLSTM

            self.link = link(
                1, 10, 10, 0.0,
                lateral_init=self.lateral_init,
                upward_init=self.upward_init,
                bias_init=self.bias_init,
                forget_bias_init=self.forget_bias_init)

    def check_param(self):
        link = self.link
        dtype = self.dtype
        for ws_i in link.ws:
            for i, w in enumerate(ws_i):
                assert w.dtype == dtype
                if 0 <= i <= 3:
                    testing.assert_allclose(w.array, self.upward_init, atol=0, rtol=0)
                elif 4 <= i <= 7:
                    testing.assert_allclose(w.array, self.lateral_init, atol=0, rtol=0)

        for bs_i in link.bs:
            for i, b in enumerate(bs_i):
                assert b.dtype == dtype
                testing.assert_allclose(b.array, self.bias_init, atol=0, rtol=0)

    def test_param_cpu(self):
        self.check_param()

    @attr.gpu
    def test_param_gpu(self):
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        self.check_param()


testing.run_module(__name__, __file__)
