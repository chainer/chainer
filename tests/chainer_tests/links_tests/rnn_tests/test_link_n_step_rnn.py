import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer import initializers
from chainer.testing import attr
from chainer.testing import condition


def relu(x):
    return x * (x > 0)


@testing.parameterize(*testing.product({
    'hidden_none': [True, False],
    'activation': ['tanh', 'relu'],
}))
class TestNStepRNN(unittest.TestCase):

    lengths = [3, 1, 2]
    n_layers = 2
    in_size = 3
    out_size = 2
    dropout = 0.0

    def setUp(self):
        shape = (self.n_layers, len(self.lengths), self.out_size)
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
        if self.activation == 'tanh':
            rnn_link_class = links.NStepRNNTanh
        elif self.activation == 'relu':
            rnn_link_class = links.NStepRNNReLU
        self.rnn = rnn_link_class(
            self.n_layers, self.in_size, self.out_size, self.dropout)

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

        with testing.assert_warns(DeprecationWarning):
            self.rnn.to_cpu()

        for batch, seq in enumerate(self.xs):
            for layer in range(self.n_layers):
                p = self.rnn[layer]
                h_prev = self.h[layer, batch]
                hs = []
                for x in seq:
                    if self.activation == 'tanh':
                        activation_func = numpy.tanh
                    elif self.activation == 'relu':
                        activation_func = relu

                    h_prev = activation_func(x.dot(p.w0.array.T) +
                                             h_prev.dot(p.w1.array.T) +
                                             p.b0.array + p.b1.array)

                    hs.append(h_prev)

                seq = hs
                testing.assert_allclose(hy.data[layer, batch], h_prev)

            for y, ey in zip(ys[batch].array, seq):
                testing.assert_allclose(y, ey)

    def test_forward_cpu_train(self):
        with chainer.using_config('train', True):
            self.check_forward(self.h, self.xs)

    @attr.gpu
    def test_forward_gpu_train(self):
        with testing.assert_warns(DeprecationWarning):
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
        with testing.assert_warns(DeprecationWarning):
            self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'always'), \
                chainer.using_config('train', False):
            self.check_forward(
                cuda.to_gpu(self.h),
                [cuda.to_gpu(x) for x in self.xs])

    def check_multi_gpu_forward(self, train=True):
        # See chainer/chainer#6262
        # NStepRNNTanh and NStepRNNReLU w/ cudnn & dropout should work on
        # not current device
        msg = None
        rnn = self.rnn.copy('copy')
        rnn.dropout = .5
        with cuda.get_device_from_id(1):
            if self.hidden_none:
                h = None
            else:
                h = cuda.to_gpu(self.h)
            xs = [cuda.to_gpu(x) for x in self.xs]
            with testing.assert_warns(DeprecationWarning):
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
            in_data = [h_data, ] + xs_data
        gradient_check.check_backward(
            fun, tuple(in_data),
            tuple([gh_data, ] + gys_data),
            tuple(params), rtol=1e-2, atol=5e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(
            self.h, self.xs, self.gh, self.gys)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        with testing.assert_warns(DeprecationWarning):
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
    'activation': ['tanh', 'relu'],
}))
class TestNStepBiRNN(unittest.TestCase):

    lengths = [3, 1, 2]
    n_layers = 2
    in_size = 3
    out_size = 2
    dropout = 0.0

    def setUp(self):
        shape = (self.n_layers * 2, len(self.lengths), self.out_size)
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
        if self.activation == 'tanh':
            rnn_link_class = links.NStepBiRNNTanh
        elif self.activation == 'relu':
            rnn_link_class = links.NStepBiRNNReLU
        self.rnn = rnn_link_class(
            self.n_layers, self.in_size, self.out_size, self.dropout)

        for layer in self.rnn:
            for p in layer.params():
                p.array[...] = numpy.random.uniform(-1, 1, p.array.shape)
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

        with testing.assert_warns(DeprecationWarning):
            self.rnn.to_cpu()

        for batch, seq in enumerate(self.xs):
            for layer in range(self.n_layers):
                # forward
                di = 0
                layer_idx = layer * 2 + di
                p = self.rnn[layer_idx]
                h_prev = self.h[layer_idx, batch]
                hs_f = []
                for x in seq:
                    if self.activation == 'tanh':
                        activation_func = numpy.tanh
                    elif self.activation == 'relu':
                        activation_func = relu

                    h_prev = activation_func(x.dot(p.w0.array.T) +
                                             h_prev.dot(p.w1.array.T) +
                                             p.b0.array + p.b1.array)
                    hs_f.append(h_prev)

                testing.assert_allclose(hy.array[layer_idx, batch], h_prev)

                # backward
                di = 1
                layer_idx = layer * 2 + di
                p = self.rnn[layer_idx]
                h_prev = self.h[layer_idx, batch]
                hs_b = []
                for x in reversed(seq):
                    if self.activation == 'tanh':
                        activation_func = numpy.tanh
                    elif self.activation == 'relu':
                        activation_func = relu
                    h_prev = activation_func(x.dot(p.w0.array.T) +
                                             h_prev.dot(p.w1.array.T) +
                                             p.b0.array + p.b1.array)
                    hs_b.append(h_prev)
                testing.assert_allclose(hy.data[layer_idx, batch], h_prev)

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
        with testing.assert_warns(DeprecationWarning):
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
        with testing.assert_warns(DeprecationWarning):
            self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'always'), \
                chainer.using_config('train', False):
            self.check_forward(
                cuda.to_gpu(self.h),
                [cuda.to_gpu(x) for x in self.xs])

    def check_multi_gpu_forward(self, train=True):
        # See chainer/chainer#6262
        # NStepBiRNNTanh and NStepBiRNNReLU w/ cudnn & dropout should work on
        # not current device
        msg = None
        rnn = self.rnn.copy('copy')
        rnn.dropout = .5
        with cuda.get_device_from_id(1):
            if self.hidden_none:
                h = None
            else:
                h = cuda.to_gpu(self.h)
            xs = [cuda.to_gpu(x) for x in self.xs]
            with testing.assert_warns(DeprecationWarning):
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
            in_data = [h_data, ] + xs_data
        gradient_check.check_backward(
            fun, tuple(in_data),
            tuple([gh_data, ] + gys_data),
            tuple(params), rtol=1e-2, atol=5e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(
            self.h, self.xs, self.gh, self.gys)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        with testing.assert_warns(DeprecationWarning):
            self.rnn.to_gpu()
        with chainer.using_config('use_cudnn', 'auto'):
            self.check_backward(
                cuda.to_gpu(self.h),
                [cuda.to_gpu(x) for x in self.xs],
                cuda.to_gpu(self.gh),
                [cuda.to_gpu(gy) for gy in self.gys])

    def test_n_cells(self):
        assert self.rnn.n_cells == 1


@testing.parameterize(
    *testing.product(
        {
            'dtype': [numpy.float32, numpy.float64],
            'initialW': ['zero', 'random'],
            'initial_bias': ['zero', 'random'],
            'activation_type': ['tanh', 'relu'],
            'use_bi_direction': [True, False]
        }
    )
)
class TestInitialization(unittest.TestCase):

    def get_initializers(self):
        if self.initialW == 'zero':
            weight_initializer = initializers.constant.Zero()
        elif self.initialW == 'random':
            weight_initializer = initializers.GlorotUniform(
                rng=numpy.random.RandomState(seed=0))

        if self.initial_bias == 'zero':
            bias_initializer = initializers.constant.Zero()
        elif self.initial_bias == 'random':
            bias_initializer = initializers.Uniform(
                rng=numpy.random.RandomState(seed=0))

        return weight_initializer, bias_initializer

    def setUp(self):
        weight_initializer, bias_initializer = self.get_initializers()
        with chainer.using_config('dtype', self.dtype):
            if self.activation_type == 'tanh':
                if self.use_bi_direction:
                    link = links.NStepBiRNNTanh
                else:
                    link = links.NStepRNNTanh

            elif self.activation_type == 'relu':
                if self.use_bi_direction:
                    link = links.NStepBiRNNReLU
                else:
                    link = links.NStepRNNReLU

            self.link = link(
                1, 10, 10, 0.0,
                initialW=weight_initializer,
                initial_bias=bias_initializer)

    def check_param(self):
        weight_initializer, bias_initializer = self.get_initializers()
        link = self.link
        xp = link.xp
        dtype = self.dtype
        for ws_i in link.ws:
            for w in ws_i:
                assert w.dtype == dtype
                w_expected = xp.empty(w.shape, dtype)
                weight_initializer(w_expected)
                testing.assert_allclose(
                    w.array, w_expected, atol=0, rtol=0)

        for bs_i in link.bs:
            for b in bs_i:
                assert b.dtype == dtype
                b_expected = xp.empty(b.shape, dtype)
                bias_initializer(b_expected)
                testing.assert_allclose(
                    b.array, b_expected, atol=0, rtol=0)

    def test_param_cpu(self):
        self.check_param()

    @attr.gpu
    def test_param_gpu(self):
        self.link.to_device('@cupy:0')
        self.check_param()


testing.run_module(__name__, __file__)
