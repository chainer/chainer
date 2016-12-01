import binascii
import os
import time

import numpy
import six

from chainer import cuda
from chainer import function
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

    def set_dropout_ratio(self, handle, dropout):
        cudnn.set_dropout_descriptor(self.desc, handle, dropout)

    @staticmethod
    def create(handle, dropout, seed):
        states = cudnn.create_dropout_states(handle)
        desc = cudnn.create_dropout_descriptor(
            handle, dropout, states.data.ptr, states.size, seed)
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
            self._states = DropoutStates.create(handle, dropout, self._seed)
        else:
            self._states.set_dropout_ratio(handle, dropout)

        return self._states


def _split(inputs, pos):
    return inputs[:pos], inputs[pos:]


if cuda.cudnn_enabled and _cudnn_version >= 5000:
    # Map string names to enums.
    # Keep enums in the same dict, so that interfaces support both.
    _rnn_dirs = {
        'uni': libcudnn.CUDNN_UNIDIRECTIONAL,
        'bi':  libcudnn.CUDNN_BIDIRECTIONAL,
        libcudnn.CUDNN_UNIDIRECTIONAL: libcudnn.CUDNN_UNIDIRECTIONAL,
        libcudnn.CUDNN_BIDIRECTIONAL: libcudnn.CUDNN_BIDIRECTIONAL,
    }

    _rnn_modes = {
        'relu': libcudnn.CUDNN_RNN_RELU,
        'tanh': libcudnn.CUDNN_RNN_TANH,
        'gru': libcudnn.CUDNN_GRU,
        'lstm': libcudnn.CUDNN_LSTM,
        libcudnn.CUDNN_RNN_RELU: libcudnn.CUDNN_RNN_RELU,
        libcudnn.CUDNN_RNN_TANH: libcudnn.CUDNN_RNN_TANH,
        libcudnn.CUDNN_GRU: libcudnn.CUDNN_GRU,
        libcudnn.CUDNN_LSTM: libcudnn.CUDNN_LSTM

    }

    _rnn_params_modes = {
        # Todo: check this is ok.
        libcudnn.CUDNN_RNN_RELU: {'n_W': 2, 'n_Wb': 2 * 2, 'n_cell': 1},
        libcudnn.CUDNN_RNN_TANH: {'n_W': 2, 'n_Wb': 2 * 2, 'n_cell': 1},
        libcudnn.CUDNN_GRU: {'n_W': 6, 'n_Wb': 2 * 6, 'n_cell': 1},
        libcudnn.CUDNN_LSTM: {'n_W': 8, 'n_Wb': 2 * 8, 'n_cell': 2},
        libcudnn.CUDNN_UNIDIRECTIONAL: {'n': 1},
        libcudnn.CUDNN_BIDIRECTIONAL: {'n': 2}
    }


class NStepRNN(function.Function):

    def __init__(self, n_layers, states, rnn_dir='uni', rnn_mode='lstm',
                 train=True):
        self.rnn_dir = _rnn_dirs[rnn_dir.lower()]
        self.rnn_mode = _rnn_modes[rnn_mode.lower()]
        self.rnn_params = _rnn_params_modes[self.rnn_mode]
        self.rnn_is_lstm_flag = self.rnn_params['n_cell'] == 2
        self.rnn_direction = _rnn_params_modes[self.rnn_dir]['n']
        self.n_layers = n_layers
        self.train = train
        self.states = states

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > self.rnn_params['n_cell'] +
                          self.rnn_params['n_Wb'] *
                          self.n_layers * self.rnn_direction)
        if self.rnn_is_lstm_flag:
            # LSTM
            (h_type, c_type), in_types = _split(in_types,
                                                self.rnn_params['n_cell'])
            type_check.expect(
                h_type.dtype == numpy.float32,
                c_type.dtype == numpy.float32,

                h_type.ndim == 3,
                h_type.shape[0] == self.n_layers * self.rnn_direction,
                c_type.ndim == 3,
                c_type.shape[0] == self.n_layers * self.rnn_direction,

                # mini-batch size
                h_type.shape[1] == c_type.shape[1],

                # hidden size
                h_type.shape[2] == c_type.shape[2],
            )
        else:
            # RNN, GRU
            (h_type, ), in_types = _split(in_types,
                                          self.rnn_params['n_cell'])
            type_check.expect(
                h_type.dtype == numpy.float32,

                h_type.ndim == 3,
                h_type.shape[0] == self.n_layers * self.rnn_direction,
            )

        w_types, in_types = _split(in_types,
                                   self.n_layers * self.rnn_direction *
                                   self.rnn_params['n_W'])
        b_types, in_types = _split(in_types,
                                   self.n_layers * self.rnn_direction *
                                   self.rnn_params['n_W'])
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
            for i in six.moves.range(self.rnn_params['n_W']):
                for di in six.moves.range(self.rnn_direction):
                    ind = (layer * self.rnn_direction +
                           di) * self.rnn_params['n_W'] + i
                    w_type = w_types[ind]
                    b_type = b_types[ind]
                    if self.rnn_direction == 1:
                        # Uni-direction
                        if layer == 0 and i < (self.rnn_params['n_W'] / 2):
                            w_in = in_size
                        else:
                            w_in = out_size
                    else:
                        # Bi-direction
                        if layer == 0 and i < 3:
                            w_in = in_size
                        elif layer > 0 and i < 3:
                            w_in = out_size * self.rnn_direction
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
        if self.rnn_is_lstm_flag:
            # LSTM
            (hx, cx), inputs = _split(inputs, self.rnn_params['n_cell'])
            cx = cuda.cupy.ascontiguousarray(cx)
            cx_desc = cudnn.create_tensor_nd_descriptor(cx)

            cy = cuda.cupy.empty_like(cx)
            cy_desc = cudnn.create_tensor_nd_descriptor(cy)

            cx_data_ptr = cx.data.ptr
            cy_data_ptr = cy.data.ptr

            cx_desc_value = cx_desc.value
            cy_desc_value = cy_desc.value
        else:
            # RNN, GRU
            (hx, ), inputs = _split(inputs, self.rnn_params['n_cell'])
            cy = None
            cx_data_ptr = 0
            cy_data_ptr = 0
            cx_desc_value = 0
            cy_desc_value = 0

        ws, inputs = _split(inputs,
                            self.n_layers *
                            self.rnn_direction * self.rnn_params['n_W'])
        bs, inputs = _split(inputs,
                            self.n_layers *
                            self.rnn_direction * self.rnn_params['n_W'])
        x_list = inputs
        hx = cuda.cupy.ascontiguousarray(hx)
        x_desc = cudnn.create_tensor_nd_descriptor(x_list[0][..., None])

        length = len(x_list)
        n_units = hx.shape[2]

        xs = cuda.cupy.concatenate(x_list, axis=0)
        ys = cuda.cupy.empty((len(xs),
                              n_units * self.rnn_direction), dtype=xs.dtype)

        handle = cudnn.get_handle()
        self.handle = handle

        rnn_desc = cudnn.create_rnn_descriptor(
            n_units, self.n_layers, self.states.desc,
            libcudnn.CUDNN_LINEAR_INPUT, self.rnn_dir,
            self.rnn_mode, libcudnn.CUDNN_DATA_FLOAT)
        self.rnn_desc = rnn_desc

        c_x_descs = _make_tensor_descriptor_array(x_list)
        hx_desc = cudnn.create_tensor_nd_descriptor(hx)

        weights_size = libcudnn.getRNNParamsSize(
            handle, rnn_desc.value, x_desc.value, libcudnn.CUDNN_DATA_FLOAT)
        w = cuda.cupy.empty((weights_size // 4, 1, 1), dtype=numpy.float32)
        w_desc = cudnn.create_filter_descriptor(w)

        for layer in six.moves.range(self.n_layers):
            for di in six.moves.range(self.rnn_direction):
                # di = 0: forward, 1: backward
                for lin_layer_id in six.moves.range(self.rnn_params['n_W']):
                    mat = cudnn.get_rnn_lin_layer_matrix_params(
                        handle, rnn_desc,
                        layer * self.rnn_direction + di,
                        x_desc, w_desc, w,
                        lin_layer_id)
                    m = mat.reshape(mat.size)
                    m[...] = ws[(layer * self.rnn_direction + di) *
                                self.rnn_params['n_W'] + lin_layer_id].ravel()
                    bias = cudnn.get_rnn_lin_layer_bias_params(
                        handle, rnn_desc,
                        layer * self.rnn_direction + di,
                        x_desc, w_desc, w,
                        lin_layer_id)
                    b = bias.reshape(bias.size)
                    b[...] = bs[(layer * self.rnn_direction + di) *
                                self.rnn_params['n_W'] + lin_layer_id]
        self.w = w
        self.w_desc = w_desc

        sections = numpy.cumsum([len(x) for x in x_list[:-1]])
        y_list = cuda.cupy.split(ys, sections)

        c_y_descs = _make_tensor_descriptor_array(y_list)
        hy = cuda.cupy.empty_like(hx)
        hy_desc = cudnn.create_tensor_nd_descriptor(hy)

        work_size = libcudnn.getRNNWorkspaceSize(
            handle, rnn_desc.value, length, c_x_descs.data)
        workspace = cuda.cupy.empty((work_size,), dtype='b')
        self.workspace = workspace

        if not self.train:
            libcudnn.RNNForwardInference(
                handle, rnn_desc.value, length,
                c_x_descs.data, xs.data.ptr, hx_desc.value, hx.data.ptr,
                cx_desc_value, cx_data_ptr, w_desc.value, w.data.ptr,
                c_y_descs.data, ys.data.ptr, hy_desc.value, hy.data.ptr,
                cy_desc_value, cy_data_ptr, workspace.data.ptr, work_size)

        else:
            reserve_size = libcudnn.getRNNTrainingReserveSize(
                handle, rnn_desc.value, length, c_x_descs.data)
            self.reserve_space = cuda.cupy.empty((reserve_size,), dtype='b')
            libcudnn.RNNForwardTraining(
                handle, rnn_desc.value, length,
                c_x_descs.data, xs.data.ptr, hx_desc.value, hx.data.ptr,
                cx_desc_value, cx_data_ptr, w_desc.value, w.data.ptr,
                c_y_descs.data, ys.data.ptr, hy_desc.value, hy.data.ptr,
                cy_desc_value, cy_data_ptr,
                workspace.data.ptr, work_size,
                self.reserve_space.data.ptr, reserve_size)

        self.c_y_descs = c_y_descs
        self.ys = ys
        self.c_x_descs = c_x_descs

        if self.rnn_is_lstm_flag:
            # LSTM
            return tuple([hy, cy] + y_list)
        else:
            # RNN, GRU
            return tuple([hy, ] + y_list)

    def backward(self, inputs, grads):
        # n_cell = 2 or 1 (LSTM or GRU, RNN)
        n_cell = self.rnn_params['n_cell']
        if self.rnn_is_lstm_flag:
            # LSTM
            (hx, cx), inputs = _split(inputs, n_cell)
            dhy, dcy = grads[:n_cell]
            if dcy is None:
                dcy = cuda.cupy.zeros_like(cx)

            cx = cuda.cupy.ascontiguousarray(cx)
            dcx = cuda.cupy.empty_like(cx)

            cx_desc = cudnn.create_tensor_nd_descriptor(cx)
            dcx_desc = cudnn.create_tensor_nd_descriptor(dcx)
            dcy_desc = cudnn.create_tensor_nd_descriptor(dcy)

            cx_data_ptr = cx.data.ptr
            dcy_data_ptr = dcy.data.ptr
            dcx_data_ptr = dcx.data.ptr
            cx_desc_value = cx_desc.value
            dcx_desc_value = dcx_desc.value
            dcy_desc_value = dcy_desc.value
        else:
            # RNN, GRU
            (hx, ), inputs = _split(inputs, n_cell)
            dhy, = grads[:n_cell]

            cx = None
            cx_data_ptr = 0
            dcy_data_ptr = 0
            dcx_data_ptr = 0
            cx_desc_value = 0
            dcx_desc_value = 0
            dcy_desc_value = 0

        ws, inputs = _split(inputs, self.n_layers * self.rnn_direction *
                            self.rnn_params['n_W'])
        bs, inputs = _split(inputs, self.n_layers * self.rnn_direction *
                            self.rnn_params['n_W'])
        x_list = inputs
        hx = cuda.cupy.ascontiguousarray(hx)

        if dhy is None:
            dhy = cuda.cupy.zeros_like(hx)

        dy_list = list(grads[n_cell:])
        for i in six.moves.range(len(dy_list)):
            if dy_list[i] is None:
                dy_list[i] = cuda.cupy.zeros_like(x_list[i])

        xs = cuda.cupy.concatenate(x_list, axis=0)
        length = len(x_list)

        dhx = cuda.cupy.empty_like(hx)

        hx_desc = cudnn.create_tensor_nd_descriptor(hx)
        dhy_desc = cudnn.create_tensor_nd_descriptor(dhy)

        c_dy_descs = _make_tensor_descriptor_array(dy_list)
        dys = cuda.cupy.concatenate(dy_list, axis=0)

        rnn_desc = self.rnn_desc
        handle = self.handle
        work_size = libcudnn.getRNNWorkspaceSize(
            handle, rnn_desc.value, length, self.c_x_descs.data)
        workspace = cuda.cupy.empty((work_size,), dtype='b')

        dhx_desc = cudnn.create_tensor_nd_descriptor(dhx)

        dxs = cuda.cupy.empty_like(xs)
        sections = numpy.cumsum([len(x) for x in x_list[:-1]])
        dx_list = cuda.cupy.split(dxs, sections, 0)
        c_dx_descs = _make_tensor_descriptor_array(dx_list)

        libcudnn.RNNBackwardData(
            handle, rnn_desc.value, length,
            self.c_y_descs.data, self.ys.data.ptr,
            c_dy_descs.data, dys.data.ptr, dhy_desc.value, dhy.data.ptr,
            dcy_desc_value, dcy_data_ptr, self.w_desc.value, self.w.data.ptr,
            hx_desc.value, hx.data.ptr, cx_desc_value, cx_data_ptr,
            c_dx_descs.data, dxs.data.ptr, dhx_desc.value, dhx.data.ptr,
            dcx_desc_value, dcx_data_ptr, workspace.data.ptr, work_size,
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
            for di in six.moves.range(self.rnn_direction):
                for lin_layer_id in six.moves.range(self.rnn_params['n_W']):
                    mat = cudnn.get_rnn_lin_layer_matrix_params(
                        handle, rnn_desc, layer * self.rnn_direction + di,
                        dx_desc, dw_desc, dw, lin_layer_id)
                    v = dws[(layer * self.rnn_direction + di) *
                            self.rnn_params['n_W'] + lin_layer_id]
                    v = v.reshape(v.size)
                    v[:] = mat.ravel()
                    bias = cudnn.get_rnn_lin_layer_bias_params(
                        handle, rnn_desc, layer * self.rnn_direction + di,
                        dx_desc, dw_desc, dw, lin_layer_id)
                    v = dbs[(layer * self.rnn_direction + di) *
                            self.rnn_params['n_W'] + lin_layer_id]
                    v = v.reshape(v.size)
                    v[:] = bias.ravel()

        if self.rnn_is_lstm_flag:
            # LSTM
            return tuple([dhx, dcx] + dws + dbs + dx_list)
        else:
            # RNN, GRU
            return tuple([dhx, ] + dws + dbs + dx_list)
