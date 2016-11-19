import binascii
import itertools
import os
import time

import numpy
import six

from chainer import cuda
from chainer import function
from chainer.functions.activation import sigmoid, tanh
from chainer.functions.array import concat
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer.functions.array import stack
from chainer.functions.connection import linear
from chainer.functions.noise import dropout
from chainer.utils import type_check


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


class NStepGRU(function.Function):

    def __init__(self, n_layers, states, train=True):
        self.n_layers = n_layers
        self.train = train
        self.states = states

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 1 + 12 * self.n_layers)
        (h_type, ), in_types = _split(in_types, 1)
        type_check.expect(
            h_type.dtype == numpy.float32,

            h_type.ndim == 3,
            h_type.shape[0] == self.n_layers,

        )
        w_types, in_types = _split(in_types, self.n_layers * 6)
        b_types, in_types = _split(in_types, self.n_layers * 6)
        x_types = in_types
        for x_type in x_types:
            type_check.expect(
                x_type.dtype == numpy.float32,
                x_type.ndim == 2,
            )
        for x1_type, x2_type in zip(x_types, x_types[1:]):
            type_check.expect(
                # Check if xs are sorted by descending lengths
                x1_type.shape[0] >= x2_type.shape[0],
                x1_type.shape[1] == x2_type.shape[1])

        in_size = x_types[0].shape[1]
        out_size = h_type.shape[2]

        for layer in six.moves.range(self.n_layers):
            for i in six.moves.range(6):
                ind = layer * 6 + i
                w_type = w_types[ind]
                b_type = b_types[ind]
                if layer == 0 and i < 3:
                    w_in = in_size
                else:
                    w_in = out_size

                type_check.expect(
                    w_type.dtype == numpy.float32,
                    w_type.ndim == 2,
                    w_type.shape[0] == out_size,
                    w_type.shape[1] == w_in,

                    b_type.dtype == numpy.float32,
                    b_type.ndim == 1,
                    b_type.shape[0] == out_size,
                )

    def forward(self, inputs):
        (hx, ), inputs = _split(inputs, 1)
        ws, inputs = _split(inputs, self.n_layers * 6)
        bs, inputs = _split(inputs, self.n_layers * 6)
        x_list = inputs

        hx = cuda.cupy.ascontiguousarray(hx)
        # Note: GRU does not need cx
        cx = cuda.cupy.zeros(hx.shape, dtype=hx.dtype)
        cx = cuda.cupy.ascontiguousarray(cx)
        x_desc = cudnn.create_tensor_nd_descriptor(x_list[0][..., None])

        length = len(x_list)
        n_units = hx.shape[2]

        xs = cuda.cupy.concatenate(x_list, axis=0)
        ys = cuda.cupy.empty((len(xs), n_units), dtype=xs.dtype)

        handle = cudnn.get_handle()
        self.handle = handle

        rnn_desc = cudnn.create_rnn_descriptor(
            n_units, self.n_layers, self.states.desc,
            libcudnn.CUDNN_LINEAR_INPUT, libcudnn.CUDNN_UNIDIRECTIONAL,
            libcudnn.CUDNN_GRU, libcudnn.CUDNN_DATA_FLOAT)
        self.rnn_desc = rnn_desc

        c_x_descs = _make_tensor_descriptor_array(x_list)
        hx_desc = cudnn.create_tensor_nd_descriptor(hx)
        cx_desc = cudnn.create_tensor_nd_descriptor(cx)
        weights_size = libcudnn.getRNNParamsSize(
            handle, rnn_desc.value, x_desc.value, libcudnn.CUDNN_DATA_FLOAT)
        w = cuda.cupy.empty((weights_size // 4, 1, 1), dtype=numpy.float32)
        w_desc = cudnn.create_filter_descriptor(w)

        for layer in six.moves.range(self.n_layers):
            for lin_layer_id in six.moves.range(6):
                mat = cudnn.get_rnn_lin_layer_matrix_params(
                    handle, rnn_desc, layer, x_desc, w_desc, w,
                    lin_layer_id)
                m = mat.reshape(mat.size)
                m[...] = ws[layer * 6 + lin_layer_id].ravel()
                bias = cudnn.get_rnn_lin_layer_bias_params(
                    handle, rnn_desc, layer, x_desc, w_desc, w,
                    lin_layer_id)
                b = bias.reshape(bias.size)
                b[...] = bs[layer * 6 + lin_layer_id]
        self.w = w
        self.w_desc = w_desc

        sections = numpy.cumsum([len(x) for x in x_list[:-1]])
        y_list = cuda.cupy.split(ys, sections)

        c_y_descs = _make_tensor_descriptor_array(y_list)
        hy = cuda.cupy.empty_like(hx)
        cy = cuda.cupy.empty_like(cx)
        hy_desc = cudnn.create_tensor_nd_descriptor(hy)
        cy_desc = cudnn.create_tensor_nd_descriptor(cy)

        work_size = libcudnn.getRNNWorkspaceSize(
            handle, rnn_desc.value, length, c_x_descs.data)
        workspace = cuda.cupy.empty((work_size,), dtype='b')
        self.workspace = workspace
        if not self.train:
            libcudnn.RNNForwardInference(
                handle, rnn_desc.value, length,
                c_x_descs.data, xs.data.ptr, hx_desc.value, hx.data.ptr,
                cx_desc.value, cx.data.ptr, w_desc.value, w.data.ptr,
                c_y_descs.data, ys.data.ptr, hy_desc.value, hy.data.ptr,
                cy_desc.value, cy.data.ptr, workspace.data.ptr, work_size)

        else:
            reserve_size = libcudnn.getRNNTrainingReserveSize(
                handle, rnn_desc.value, length, c_x_descs.data)
            self.reserve_space = cuda.cupy.empty((reserve_size,), dtype='b')
            libcudnn.RNNForwardTraining(
                handle, rnn_desc.value, length,
                c_x_descs.data, xs.data.ptr, hx_desc.value, hx.data.ptr,
                cx_desc.value, cx.data.ptr, w_desc.value, w.data.ptr,
                c_y_descs.data, ys.data.ptr, hy_desc.value, hy.data.ptr,
                cy_desc.value, cy.data.ptr,
                workspace.data.ptr, work_size,
                self.reserve_space.data.ptr, reserve_size)

        self.c_y_descs = c_y_descs
        self.ys = ys
        self.c_x_descs = c_x_descs
        return tuple([hy, ] + y_list)

    def backward(self, inputs, grads):
        (hx, ), inputs = _split(inputs, 1)
        ws, inputs = _split(inputs, self.n_layers * 6)
        bs, inputs = _split(inputs, self.n_layers * 6)
        x_list = inputs

        hx = cuda.cupy.ascontiguousarray(hx)
        # Note: GRU does not need cx
        cx = cuda.cupy.zeros(hx.shape, dtype=hx.dtype)
        cx = cuda.cupy.ascontiguousarray(cx)

        dhy = grads[0]
        dy_list = list(grads[1:])
        if dhy is None:
            dhy = cuda.cupy.zeros_like(hx)
        dcy = cuda.cupy.zeros_like(cx)

        for i in six.moves.range(len(dy_list)):
            if dy_list[i] is None:
                dy_list[i] = cuda.cupy.zeros_like(x_list[i])

        xs = cuda.cupy.concatenate(x_list, axis=0)
        length = len(x_list)

        dhx = cuda.cupy.empty_like(hx)
        dcx = cuda.cupy.empty_like(cx)

        hx_desc = cudnn.create_tensor_nd_descriptor(hx)
        cx_desc = cudnn.create_tensor_nd_descriptor(hx)
        dhy_desc = cudnn.create_tensor_nd_descriptor(dhy)
        dcy_desc = cudnn.create_tensor_nd_descriptor(dcy)

        c_dy_descs = _make_tensor_descriptor_array(dy_list)
        dys = cuda.cupy.concatenate(dy_list, axis=0)

        rnn_desc = self.rnn_desc
        handle = self.handle
        work_size = libcudnn.getRNNWorkspaceSize(
            handle, rnn_desc.value, length, self.c_x_descs.data)
        workspace = cuda.cupy.empty((work_size,), dtype='b')

        dhx_desc = cudnn.create_tensor_nd_descriptor(dhx)
        dcx_desc = cudnn.create_tensor_nd_descriptor(dcx)

        dxs = cuda.cupy.empty_like(xs)
        sections = numpy.cumsum([len(x) for x in x_list[:-1]])
        dx_list = cuda.cupy.split(dxs, sections, 0)
        c_dx_descs = _make_tensor_descriptor_array(dx_list)

        libcudnn.RNNBackwardData(
            handle, rnn_desc.value, length,
            self.c_y_descs.data, self.ys.data.ptr,
            c_dy_descs.data, dys.data.ptr, dhy_desc.value, dhy.data.ptr,
            dcy_desc.value, dcy.data.ptr, self.w_desc.value, self.w.data.ptr,
            hx_desc.value, hx.data.ptr, cx_desc.value, hx.data.ptr,
            c_dx_descs.data, dxs.data.ptr, dhx_desc.value, dhx.data.ptr,
            dcx_desc.value, dcx.data.ptr, workspace.data.ptr, work_size,
            self.reserve_space.data.ptr, self.reserve_space.size)

        dw = cuda.cupy.zeros_like(self.w)
        dw_desc = cudnn.create_tensor_nd_descriptor(dw)
        libcudnn.RNNBackwardWeights(
            handle, rnn_desc.value, length,
            self.c_x_descs.data, xs.data.ptr,
            hx_desc.value, hx.data.ptr, self.c_y_descs.data, self.ys.data.ptr,
            workspace.data.ptr, work_size, dw_desc.value, dw.data.ptr,
            self.reserve_space.data.ptr, self.reserve_space.size)

        dx = dx_list[0]
        dx = dx.reshape(dx.shape + (1,))
        dx_desc = cudnn.create_tensor_nd_descriptor(dx)
        dws = [cuda.cupy.empty_like(w) for w in ws]
        dbs = [cuda.cupy.empty_like(b) for b in bs]
        for layer in six.moves.range(self.n_layers):
            for lin_layer_id in six.moves.range(6):
                mat = cudnn.get_rnn_lin_layer_matrix_params(
                    handle, rnn_desc, layer, dx_desc, dw_desc, dw,
                    lin_layer_id)
                v = dws[layer * 6 + lin_layer_id]
                v = v.reshape(v.size)
                v[:] = mat.ravel()
                bias = cudnn.get_rnn_lin_layer_bias_params(
                    handle, rnn_desc, layer, dx_desc, dw_desc, dw,
                    lin_layer_id)
                v = dbs[layer * 6 + lin_layer_id]
                v = v.reshape(v.size)
                v[:] = bias.ravel()

        return tuple([dhx] + dws + dbs + dx_list)


def _stack_weight(ws):
    # TODO(unno): Input of the current LSTM implementaiton is shuffled
    w = stack.stack(ws, axis=1)
    shape = w.data.shape
    return reshape.reshape(w, (shape[0] * shape[1],) + shape[2:])


def n_step_gru(
        n_layers, dropout_ratio, hx, ws, bs, xs, train=True,
        use_cudnn=True):

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
        """
        From Nvidia documents:
        GRU
        - Values 0 and 3 reference the reset gate.
        - Values 1 and 4 reference the update gate.
        - Values 2 and 5 reference the new memory gate.

        rt = sigmoid(Wr xt + Rr ht-1 + bWr + bRr)
        it = sigmoid(Wi xt + Ri ht-1 + bWi + bRu)
        h't = tanh(Wh xt + rt dot (Rh ht-1 + bRh) + bWh)
        ht = (1 - it) dot h't + it dot ht-1

        it, rt, h't represent the input, reset, new gates respectively.

        """
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
