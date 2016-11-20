import binascii
import itertools
import os
import time

import numpy
import six

from chainer import cuda
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import concat
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer.functions.array import stack
from chainer.functions.connection import linear
from chainer.functions.connection.n_step_lstm import NStepRNN
from chainer.functions.noise import dropout


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()


class PointerArray(object):

    def __init__(self, lst, back_pointer):
        self._value = numpy.array(lst, dtype=numpy.intp)
        # Store back_pointer to prevent the GC removes the original variable
        self._back_pointer = back_pointer

    @property
    def data(self):
        return self._value.ctypes.data


def _make_tensor_descriptor_array(xs):
    """Make an array of pointers denoting pointers of tensor descriptors.

    """
    descs = []
    for x in xs:
        if x.ndim < 3:
            shape = x.shape + (1,) * (3 - x.ndim)
            x = x.reshape(shape)
        desc = cudnn.create_tensor_nd_descriptor(x)
        descs.append(desc)
    return PointerArray([d.value for d in descs], descs)


def _make_ptr_array(xs):
    """Make an array of pointers denoting pointers of ndarrays.

    """
    return PointerArray([x.data.ptr for x in xs], xs)


class DropoutStates(object):

    def __init__(self, states, desc):
        self.states = states
        self.desc = desc

    @staticmethod
    def create(handle, states, dropout, seed):
        desc = cudnn.create_dropout_descriptor(
            handle, dropout, states.data.ptr, states.size, seed)
        return DropoutStates(states, desc)

    @staticmethod
    def from_states(handle, states, dropout):
        desc = cudnn.create_dropout_descriptor(handle, dropout, 0, 0, 0)
        return DropoutStates(states, desc)


class DropoutRandomStates(object):

    def __init__(self, seed):
        self._states = None

        if seed is None:
            try:
                seed_str = binascii.hexlify(os.urandom(8))
                seed = numpy.uint64(int(seed_str, 16))
            except NotImplementedError:
                seed = numpy.uint64(time.clock() * 1000000)
        else:
            seed = numpy.uint64(seed)

        self._seed = seed

    def create_dropout_states(self, dropout):
        handle = cudnn.get_handle()
        if self._states is None:
            self._states = cudnn.create_dropout_states(handle)
            return DropoutStates.create(
                handle, self._states, dropout, self._seed)
        else:
            return DropoutStates.from_states(handle, self._states, dropout)


_random_states = {}


def get_random_state():
    global _random_states
    dev = cuda.Device()
    rs = _random_states.get(dev.id, None)
    if rs is None:
        rs = DropoutRandomStates(os.getenv('CHAINER_SEED'))
        _random_states[dev.id] = rs
    return rs


def _split(inputs, pos):
    return inputs[:pos], inputs[pos:]


class NStepGRU(NStepRNN):
    def __init__(self, n_layers, states, train=True):
        NStepRNN.__init__(self, n_layers, states, rnn_dir='uni',
                          rnn_mode='gru', train=train)


class NStepBiGRU(NStepRNN):
    def __init__(self, n_layers, states, train=True):
        NStepRNN.__init__(self, n_layers, states, rnn_dir='bi',
                          rnn_mode='gru', train=train)


def _stack_weight(ws):
    # TODO(unno): Input of the current LSTM implementaiton is shuffled
    w = stack.stack(ws, axis=1)
    shape = w.data.shape
    return reshape.reshape(w, (shape[0] * shape[1],) + shape[2:])


def n_step_gru(
        n_layers, dropout_ratio, hx, ws, bs, xs, train=True,
        use_cudnn=True):
    """Stacked GRU function for sequence inputs. Todo: write document."""

    xp = cuda.get_array_module(hx.data)

    if use_cudnn and xp is not numpy and cuda.cudnn_enabled and \
       _cudnn_version >= 5000:
        # CUDNN version
        states = get_random_state().create_dropout_states(dropout_ratio)
        # flatten all input variables
        inputs = tuple(itertools.chain(
            (hx, ),
            itertools.chain.from_iterable(ws),
            itertools.chain.from_iterable(bs),
            xs))
        rnn = NStepGRU(n_layers, states, train=train)
        ret = rnn(*inputs)
        hy = ret[0]
        ys = ret[1:]
        return hy, ys

    else:
        hx = split_axis.split_axis(hx, n_layers, axis=0, force_tuple=True)
        hx = [reshape.reshape(h, h.data.shape[1:]) for h in hx]
        # From Nvidia documents:
        # GRU
        # - Values 0 and 3 reference the reset gate.
        # - Values 1 and 4 reference the update gate.
        # - Values 2 and 5 reference the new memory gate.
        xws = [concat.concat([w[0], w[1], w[2]], axis=0) for w in ws]
        hws = [concat.concat([w[3], w[4], w[5]], axis=0) for w in ws]
        xbs = [concat.concat([b[0], b[1], b[2]], axis=0) for b in bs]
        hbs = [concat.concat([b[3], b[4], b[5]], axis=0) for b in bs]

        ys = []
        for x in xs:
            batch = x.shape[0]
            h_next = []
            for layer in six.moves.range(n_layers):
                h = hx[layer]
                if h.shape[0] > batch:
                    h, h_rest = split_axis.split_axis(h, [batch], axis=0)
                else:
                    h_rest = None

                x = dropout.dropout(x, ratio=dropout_ratio, train=train)
                h = dropout.dropout(h, ratio=dropout_ratio, train=train)

                gru_in_x = linear.linear(x, xws[layer], xbs[layer])
                gru_in_h = linear.linear(h, hws[layer], hbs[layer])
                W_r_x, W_z_x, W_x = split_axis.split_axis(gru_in_x, 3, axis=1)
                U_r_h, U_z_h, U_x = split_axis.split_axis(gru_in_h, 3, axis=1)

                r = sigmoid.sigmoid(W_r_x + U_r_h)
                z = sigmoid.sigmoid(W_z_x + U_z_h)
                h_bar = tanh.tanh(W_x + r*U_x)
                h_bar = (1 - z) * h_bar + z * h

                if h_rest is not None:
                    h = concat.concat([h_bar, h_rest], axis=0)
                else:
                    h = h_bar

                h_next.append(h)
                x = h_bar
            hx = h_next
            ys.append(x)

        hy = stack.stack(hx)
        return hy, tuple(ys)


def n_step_bigru(n_layers, dropout_ratio, hx, ws, bs, xs, train=True,
                 use_cudnn=True):
    """Bi-GRU function. Todo: write document. """

    xp = cuda.get_array_module(hx, hx.data)

    if use_cudnn and xp is not numpy and cuda.cudnn_enabled and \
       _cudnn_version >= 5000:
        states = get_random_state().create_dropout_states(dropout_ratio)
        # flatten all input variables
        inputs = tuple(itertools.chain(
            (hx,),
            itertools.chain.from_iterable(ws),
            itertools.chain.from_iterable(bs),
            xs))
        rnn = NStepBiGRU(n_layers, states, train=train)
        ret = rnn(*inputs)
        hy, = ret[:1]
        ys = ret[1:]
        return hy, ys

    else:
        hx = split_axis.split_axis(hx, n_layers * 2, axis=0, force_tuple=True)
        hx = [reshape.reshape(h, h.shape[1:]) for h in hx]

        # xws = [_stack_weight([w[0], w[1], w[2]]) for w in ws]
        # hws = [_stack_weight([w[3], w[4], w[5]]) for w in ws]
        # xbs = [_stack_weight([b[0], b[1], b[2]]) for b in bs]
        # hbs = [_stack_weight([b[3], b[4], b[5]]) for b in bs]

        xws = [concat.concat([w[0], w[1], w[2]], axis=0) for w in ws]
        hws = [concat.concat([w[3], w[4], w[5]], axis=0) for w in ws]
        xbs = [concat.concat([b[0], b[1], b[2]], axis=0) for b in bs]
        hbs = [concat.concat([b[3], b[4], b[5]], axis=0) for b in bs]

        batches = [x.shape[0] for x in xs]
        hy = []
        _xs = xs
        for layer in range(n_layers):
            # forward
            di = 0
            h = hx[2 * layer + di]
            hf = []
            for batch, x in zip(batches, _xs):
                if h.shape[0] > batch:
                    h, h_rest = split_axis.split_axis(h, [batch], axis=0)
                else:
                    h_rest = None
                if x.shape[0] > batch:
                    x, _ = split_axis.split_axis(x, [batch], axis=0)

                x = dropout.dropout(x, ratio=dropout_ratio, train=train)
                h = dropout.dropout(h, ratio=dropout_ratio, train=train)
                gru_in_x = linear.linear(x, xws[2*layer+di], xbs[2*layer+di])
                gru_in_h = linear.linear(h, hws[2*layer+di], hbs[2*layer+di])
                r_x, z_x, k_x = split_axis.split_axis(gru_in_x, 3, axis=1)
                r_h, z_h, k_h = split_axis.split_axis(gru_in_h, 3, axis=1)
                r = sigmoid.sigmoid(r_x + r_h)
                z = sigmoid.sigmoid(z_x + z_h)
                h_bar = z * h + (1 - z) * tanh.tanh(k_x + r * k_h)

                if h_rest is not None:
                    h = concat.concat([h_bar, h_rest], axis=0)
                else:
                    h = h_bar
                hf.append(h)
            hy.append(h)

            # backward
            di = 1
            h = hx[2 * layer + di]
            hb = []
            for batch, x in reversed(list(zip(batches, _xs))):
                if h.shape[0] > batch:
                    h, h_rest = split_axis.split_axis(h, [batch], axis=0)
                else:
                    h_rest = None
                if x.shape[0] > batch:
                    x, _ = split_axis.split_axis(x, [batch], axis=0)

                x = dropout.dropout(x, ratio=dropout_ratio, train=train)
                h = dropout.dropout(h, ratio=dropout_ratio, train=train)
                gru_in_x = linear.linear(x, xws[2*layer+di], xbs[2*layer+di])
                gru_in_h = linear.linear(h, hws[2*layer+di], hbs[2*layer+di])
                r_x, z_x, k_x = split_axis.split_axis(gru_in_x, 3, axis=1)
                r_h, z_h, k_h = split_axis.split_axis(gru_in_h, 3, axis=1)
                r = sigmoid.sigmoid(r_x + r_h)
                z = sigmoid.sigmoid(z_x + z_h)
                h_bar = z * h + (1 - z) * tanh.tanh(k_x + r * k_h)

                if h_rest is not None:
                    h = concat.concat([h_bar, h_rest], axis=0)
                else:
                    h = h_bar
                hb.append(h)
            hy.append(h)
            hb.reverse()
            _xs = [concat.concat([hfi, hbi], axis=1)
                   for (hfi, hbi) in zip(hf, hb)]

        hy = stack.stack(hy)
        ys = [_x[:batch, :] for (batch, _x) in zip(batches, _xs)]
        return hy, tuple(ys)
