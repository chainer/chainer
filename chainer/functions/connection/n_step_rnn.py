import itertools

import numpy
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import configuration
from chainer import function
from chainer.functions.activation import relu
from chainer.functions.activation import tanh
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer.functions.array import stack
from chainer.functions.connection import linear
from chainer.functions.noise import dropout
from chainer.utils import argument
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.libcudnn
    _cudnn_version = libcudnn.getVersion()


if cuda.cudnn_enabled and _cudnn_version >= 5000:
    # Define RNN parameters using dict.
    _rnn_dirs = {
        'uni': libcudnn.CUDNN_UNIDIRECTIONAL,
        'bi':  libcudnn.CUDNN_BIDIRECTIONAL,
    }

    _rnn_modes = {
        'rnn_relu': libcudnn.CUDNN_RNN_RELU,
        'rnn_tanh': libcudnn.CUDNN_RNN_TANH,
        'gru': libcudnn.CUDNN_GRU,
        'lstm': libcudnn.CUDNN_LSTM,
    }

    _rnn_n_params = {
        libcudnn.CUDNN_RNN_RELU: 2,
        libcudnn.CUDNN_RNN_TANH: 2,
        libcudnn.CUDNN_GRU: 6,
        libcudnn.CUDNN_LSTM: 8,
    }

    _rnn_params_direction = {
        libcudnn.CUDNN_UNIDIRECTIONAL: 1,
        libcudnn.CUDNN_BIDIRECTIONAL: 2,
    }

    _rnn_params_use_cell = {
        libcudnn.CUDNN_RNN_RELU: False,
        libcudnn.CUDNN_RNN_TANH: False,
        libcudnn.CUDNN_GRU: False,
        libcudnn.CUDNN_LSTM: True,
    }


class CudnnRNNWeightConcat(function.Function):

    """Concatenates weight matrices for cuDNN's RNN.

    This function concatenates weight matrices for RNNs into one large array.
    Its format is defined in cuDNN's API.

    """

    def __init__(self, n_layers, states, rnn_dir, rnn_mode):
        self.n_layers = n_layers
        self.states = states
        self.rnn_dir = _rnn_dirs[rnn_dir]
        self.rnn_mode = _rnn_modes[rnn_mode]

        self.rnn_direction = _rnn_params_direction[self.rnn_dir]
        self.n_W = _rnn_n_params[self.rnn_mode]

    def check_type_forward(self, in_types):
        n_params = self.n_layers * self.rnn_direction * self.n_W
        type_check.expect(
            in_types.size() == n_params * 2)

        w_types = in_types[:n_params]
        b_types = in_types[n_params:]

        in_size = w_types[0].shape[1]
        out_size = w_types[0].shape[0]

        for layer in six.moves.range(self.n_layers):
            for di in six.moves.range(self.rnn_direction):
                for i in six.moves.range(self.n_W):
                    ind = (layer * self.rnn_direction + di) * self.n_W + i
                    w_type = w_types[ind]
                    b_type = b_types[ind]
                    if self.rnn_direction == 1:
                        # Uni-direction
                        if layer == 0 and i < (self.n_W // 2):
                            w_in = in_size
                        else:
                            w_in = out_size
                    else:
                        # Bi-direction
                        if layer == 0 and i < (self.n_W // 2):
                            w_in = in_size
                        elif layer > 0 and i < (self.n_W // 2):
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

    def forward_gpu(self, inputs):
        handle = cudnn.get_handle()
        ws_size = self.n_layers * self.rnn_direction * self.n_W
        ws = inputs[:ws_size]
        bs = inputs[ws_size:]
        out_size = ws[0].shape[0]
        in_size = ws[0].shape[1]

        # TODO(unno): Make a wrapper method to avoid access _desc directly
        rnn_desc = cudnn.create_rnn_descriptor(
            out_size, self.n_layers, self.states._desc,
            libcudnn.CUDNN_LINEAR_INPUT, self.rnn_dir,
            self.rnn_mode, libcudnn.CUDNN_DATA_FLOAT)
        self.rnn_desc = rnn_desc

        dummy_x = cuda.cupy.empty((1, in_size, 1), numpy.float32)
        x_desc = cudnn.create_tensor_nd_descriptor(dummy_x)

        weights_size = libcudnn.getRNNParamsSize(
            handle, rnn_desc.value, x_desc.value, libcudnn.CUDNN_DATA_FLOAT)
        w = cuda.cupy.empty((weights_size // 4, 1, 1), dtype=numpy.float32)
        w_desc = cudnn.create_filter_descriptor(w)

        for layer in six.moves.range(self.n_layers):
            for di in six.moves.range(self.rnn_direction):
                mat_index = layer * self.rnn_direction + di
                # di = 0: forward, 1: backward
                for lin_layer_id in six.moves.range(self.n_W):
                    mat = cudnn.get_rnn_lin_layer_matrix_params(
                        handle, rnn_desc, mat_index,
                        x_desc, w_desc, w, lin_layer_id)
                    W_index = mat_index * self.n_W + lin_layer_id
                    m = mat.reshape(mat.size)
                    m[...] = ws[W_index].ravel()
                    bias = cudnn.get_rnn_lin_layer_bias_params(
                        handle, rnn_desc, mat_index,
                        x_desc, w_desc, w, lin_layer_id)
                    b = bias.reshape(bias.size)
                    b[...] = bs[W_index]
        self.w_desc = w_desc
        self.x_desc = x_desc
        return w,

    def backward(self, inputs, grads):
        handle = cudnn.get_handle()
        ws_size = self.n_layers * self.rnn_direction * self.n_W
        ws = inputs[0:ws_size]
        bs = inputs[ws_size:]

        rnn_desc = self.rnn_desc
        dw = grads[0]
        dw_desc = cudnn.create_filter_descriptor(dw)
        dx_desc = self.x_desc

        dws = []
        dbs = []
        for layer in six.moves.range(self.n_layers):
            for di in six.moves.range(self.rnn_direction):
                mat_index = layer * self.rnn_direction + di
                for lin_layer_id in six.moves.range(self.n_W):
                    mat = cudnn.get_rnn_lin_layer_matrix_params(
                        handle, rnn_desc, mat_index,
                        dx_desc, dw_desc, dw, lin_layer_id)
                    W_index = mat_index * self.n_W + lin_layer_id
                    dws.append(mat.reshape(ws[W_index].shape))
                    bias = cudnn.get_rnn_lin_layer_bias_params(
                        handle, rnn_desc, mat_index,
                        dx_desc, dw_desc, dw, lin_layer_id)
                    dbs.append(bias.reshape(bs[W_index].shape))
        return tuple(dws + dbs)


def cudnn_rnn_weight_concat(
        n_layers, states, use_bi_direction, rnn_mode, ws, bs):
    rnn_dir = 'bi' if use_bi_direction else 'uni'
    inputs = itertools.chain(
        itertools.chain.from_iterable(ws),
        itertools.chain.from_iterable(bs),
    )
    return CudnnRNNWeightConcat(n_layers, states, rnn_dir, rnn_mode)(*inputs)


class BaseNStepRNN(function.Function):

    def __init__(self, n_layers, states, lengths, rnn_dir, rnn_mode, **kwargs):
        if kwargs:
            argument.check_unexpected_kwargs(
                kwargs, train='train argument is not supported anymore. '
                'Use chainer.using_config')
            argument.assert_kwargs_empty(kwargs)

        if rnn_dir not in _rnn_dirs:
            candidate_list = ','.join(_rnn_dirs.keys())
            raise ValueError('Invalid rnn_dir: "%s". Please select from [%s]'
                             % (rnn_dir, candidate_list))
        if rnn_mode not in _rnn_modes:
            candidate_list = ','.join(_rnn_modes.keys())
            raise ValueError('Invalid rnn_mode: "%s". Please select from [%s]'
                             % (rnn_mode, candidate_list))
        self.rnn_dir = _rnn_dirs[rnn_dir]
        self.rnn_mode = _rnn_modes[rnn_mode]
        self.rnn_direction = _rnn_params_direction[self.rnn_dir]
        self.n_layers = n_layers
        self.states = states
        self.use_cell = _rnn_params_use_cell[self.rnn_mode]
        self.lengths = lengths
        self.sections = numpy.cumsum(lengths)

    def check_type_forward(self, in_types):
        if self.use_cell:
            type_check.expect(in_types.size() == 4)
            h_type, c_type, w_type, x_type = in_types
            h_size = self.n_layers * self.rnn_direction
            type_check.expect(
                h_type.dtype == numpy.float32,
                c_type.dtype == numpy.float32,

                h_type.ndim == 3,
                h_type.shape[0] == h_size,
                c_type.ndim == 3,
                c_type.shape[0] == h_size,

                # mini-batch size
                h_type.shape[1] == c_type.shape[1],

                # hidden size
                h_type.shape[2] == c_type.shape[2],
            )

        else:
            type_check.expect(in_types.size() == 3)
            h_type, w_type, x_type = in_types
            h_size = self.n_layers * self.rnn_direction
            type_check.expect(
                h_type.dtype == numpy.float32,

                h_type.ndim == 3,
                h_type.shape[0] == h_size,
            )

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 2,
            x_type.shape[0] == self.sections[-1],
        )

    def forward_gpu(self, inputs):
        if self.use_cell:
            # LSTM
            hx, cx, w, xs = inputs
        else:
            # RNN, GRU
            hx, w, xs = inputs
            cx = None

        if not configuration.config.train:
            hy, cy, ys = cudnn.rnn_forward_inference(
                self.states, self.rnn_dir, self.rnn_mode,
                hx, cx, w, xs, self.lengths)

        else:
            self.reserve_space, hy, cy, ys = cudnn.rnn_forward_training(
                self.states, self.rnn_dir, self.rnn_mode,
                hx, cx, w, xs, self.lengths)

        if self.use_cell:
            # LSTM
            self.retain_outputs((2,))
            return hy, cy, ys
        else:
            # GRU, RNN
            self.retain_outputs((1,))
            return hy, ys

    def backward(self, inputs, grads):
        if not configuration.config.train:
            raise RuntimeError('cuDNN does not support backward computation '
                               'of RNN in testing mode')

        if self.use_cell:
            # LSTM
            hx, cx, w, xs = inputs
            dhy, dcy, dys = grads
            if dcy is None:
                dcy = cuda.cupy.zeros_like(cx)

        else:
            # GRU, RNN
            hx, w, xs = inputs
            dhy, dys = grads
            dcy = cx = None

        ys = self.output_data[-1]

        if dhy is None:
            dhy = cuda.cupy.zeros_like(hx)
        if dys is None:
            dys = cuda.cupy.zeros_like(ys)

        dhx, dcx, dxs = cudnn.rnn_backward_data(
            self.states, self.rnn_dir, self.rnn_mode,
            hx, cx, w, xs, ys, self.reserve_space,
            dhy, dcy, dys, self.lengths)

        dw = cudnn.rnn_backward_weights(
            self.states, self.rnn_dir, self.rnn_mode,
            xs, hx, ys, w, self.reserve_space, self.lengths)

        if self.use_cell:
            # LSTM
            return dhx, dcx, dw, dxs
        else:
            # GRU, RNN
            return dhx, dw, dxs


class NStepRNNTanh(BaseNStepRNN):

    def __init__(self, n_layers, states, lengths, **kwargs):
        BaseNStepRNN.__init__(
            self, n_layers, states, lengths,
            rnn_dir='uni', rnn_mode='rnn_tanh', **kwargs)


class NStepRNNReLU(BaseNStepRNN):

    def __init__(self, n_layers, states, lengths, **kwargs):
        BaseNStepRNN.__init__(
            self, n_layers, states, lengths,
            rnn_dir='uni', rnn_mode='rnn_relu', **kwargs)


class NStepBiRNNTanh(BaseNStepRNN):

    def __init__(self, n_layers, states, lengths, **kwargs):
        BaseNStepRNN.__init__(
            self, n_layers, states, lengths,
            rnn_dir='bi', rnn_mode='rnn_tanh', **kwargs)


class NStepBiRNNReLU(BaseNStepRNN):

    def __init__(self, n_layers, states, lengths, **kwargs):
        BaseNStepRNN.__init__(
            self, n_layers, states, lengths,
            rnn_dir='bi', rnn_mode='rnn_relu', **kwargs)


def n_step_rnn(
        n_layers, dropout_ratio, hx, ws, bs, xs, activation='tanh', **kwargs):
    """n_step_rnn(n_layers, dropout_ratio, hx, ws, bs, xs, activation='tanh')

    Stacked Uni-directional RNN function for sequence inputs.

    This function calculates stacked Uni-directional RNN with sequences.
    This function gets an initial hidden state :math:`h_0`,
    an initial cell state :math:`c_0`, an input sequence :math:`x`,
    weight matrices :math:`W`, and bias vectors :math:`b`.
    This function calculates hidden states :math:`h_t` and :math:`c_t` for each
    time :math:`t` from input :math:`x_t`.

    .. math::
       h_t = f(W_0 x_t + W_1 h_{t-1} + b_0 + b_1)

    where :math:`f` is an activation function.

    Weight matrices :math:`W` contains two matrices :math:`W_0` and
    :math:`W_1`. :math:`W_0` is a parameter for an input sequence.
    :math:`W_1` is a parameter for a hidden state.
    Bias matrices :math:`b` contains two matrices :math:`b_0` and :math:`b_1`.
    :math:`b_0` is a parameter for an input sequence.
    :math:`b_1` is a parameter for a hidden state.


    As the function accepts a sequence, it calculates :math:`h_t` for all
    :math:`t` with one call. Two weight matrices and two bias vectors are
    required for each layer. So, when :math:`S` layers exist, you need to
    prepare :math:`2S` weight matrices and :math:`2S` bias vectors.

    If the number of layers ``n_layers`` is greather than :math:`1`, input
    of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.
    Note that all input variables except first layer may have different shape
    from the first layer.

    Args:
        n_layers(int): Number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (:class:`~chainer.Variable`):
            Variable holding stacked hidden states.
            Its shape is ``(S, B, N)`` where ``S`` is number of layers and is
            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
            dimension of hidden units.
        ws (list of list of :class:`~chainer.Variable`): Weight matrices.
            ``ws[i]`` represents weights for i-th layer.
            Each ``ws[i]`` is a list containing two matrices.
            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 1`` is ``(I, N)`` shape as they
            are multiplied with input variables. All other matrices has
            ``(N, N)`` shape.
        bs (list of list of :class:`~chainer.Variable`): Bias vectors.
            ``bs[i]`` represnents biases for i-th layer.
            Each ``bs[i]`` is a list containing two vectors.
            ``bs[i][j]`` is corresponding with ``b_j`` in the equation.
            Shape of each matrix is ``(N,)`` where ``N`` is dimension of
            hidden units.
        xs (list of :class:`~chainer.Variable`):
            A list of :class:`~chainer.Variable` holding input values.
            Each element ``xs[t]`` holds input value for time ``t``.
            Its shape is ``(B_t, I)``, where ``B_t`` is
            mini-batch size for time ``t``, and ``I`` is size of input units.
            Note that this function supports variable length sequences.
            When sequneces has different lengths, sort sequences in descending
            order by length, and transpose the sorted sequence.
            :func:`~chainer.functions.transpose_sequence` transpose a list
            of :func:`~chainer.Variable` holding sequence.
            So ``xs`` needs to satisfy
            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.
        activation (str): Activation function name.
            Please select ``tanh`` or ``relu``.

    Returns:
        tuple: This function returns a tuple containing three elements,
        ``hy`` and ``ys``.

        - ``hy`` is an updated hidden states whose shape is same as ``hx``.
        - ``ys`` is a list of :class:`~chainer.Variable` . Each element
          ``ys[t]`` holds hidden states of the last layer corresponding
          to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is
          mini-batch size for time ``t``, and ``N`` is size of hidden
          units. Note that ``B_t`` is the same value as ``xs[t]``.

    """
    return n_step_rnn_base(n_layers, dropout_ratio, hx, ws, bs, xs,
                           activation, use_bi_direction=False, **kwargs)


def n_step_birnn(
        n_layers, dropout_ratio, hx, ws, bs, xs, activation='tanh', **kwargs):
    """n_step_birnn(n_layers, dropout_ratio, hx, ws, bs, xs, activation='tanh')

    Stacked Bi-directional RNN function for sequence inputs.

    This function calculates stacked Bi-directional RNN with sequences.
    This function gets an initial hidden state :math:`h_0`, an initial
    cell state :math:`c_0`, an input sequence :math:`x`,
    weight matrices :math:`W`, and bias vectors :math:`b`.
    This function calculates hidden states :math:`h_t` and :math:`c_t` for each
    time :math:`t` from input :math:`x_t`.

    .. math::
        h^{f}_t &=& f(W^{f}_0 x_t + W^{f}_1 h_{t-1} + b^{f}_0 + b^{f}_1), \\\\
        h^{b}_t &=& f(W^{b}_0 x_t + W^{b}_1 h_{t-1} + b^{b}_0 + b^{b}_1), \\\\
        h_t  &=& [h^{f}_t; h^{f}_t], \\\\

    where :math:`f` is an activation function.

    Weight matrices :math:`W` contains two matrices :math:`W^{f}` and
    :math:`W^{b}`. :math:`W^{f}` is weight matrices for forward directional
    RNN. :math:`W^{b}` is weight matrices for backward directional RNN.

    :math:`W^{f}` contains :math:`W^{f}_0` for an input sequence and
    :math:`W^{f}_1` for a hidden state.
    :math:`W^{b}` contains :math:`W^{b}_0` for an input sequence and
    :math:`W^{b}_1` for a hidden state.

    Bias matrices :math:`b` contains two matrices :math:`b^{f}` and
    :math:`b^{f}`. :math:`b^{f}` contains :math:`b^{f}_0` for an input sequence
    and :math:`b^{f}_1` for a hidden state.
    :math:`b^{b}` contains :math:`b^{b}_0` for an input sequence and
    :math:`b^{b}_1` for a hidden state.

    As the function accepts a sequence, it calculates :math:`h_t` for all
    :math:`t` with one call. Two weight matrices and two bias vectors are
    required for each layer. So, when :math:`S` layers exist, you need to
    prepare :math:`2S` weight matrices and :math:`2S` bias vectors.

    If the number of layers ``n_layers`` is greather than :math:`1`, input
    of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.
    Note that all input variables except first layer may have different shape
    from the first layer.

    Args:
        n_layers(int): Number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (:class:`~chainer.Variable`):
            Variable holding stacked hidden states.
            Its shape is ``(2S, B, N)`` where ``S`` is number of layers and is
            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
            dimension of hidden units. Because of bi-direction, the
            first dimension length is ``2S``.
        ws (list of list of :class:`~chainer.Variable`): Weight matrices.
            ``ws[i + di]`` represents weights for i-th layer.
            Note that ``di = 0`` for forward-RNN and ``di = 1`` for
            backward-RNN.
            Each ``ws[i + di]`` is a list containing two matrices.
            ``ws[i + di][j]`` is corresponding with ``W^{f}_j`` if ``di = 0``
            and corresponding with ``W^{b}_j`` if ``di = 1`` in the equation.
            Only ``ws[0][j]`` and ``ws[1][j]`` where ``0 <= j < 1`` are
            ``(I, N)`` shape as they are multiplied with input variables.
            All other matrices has ``(N, N)`` shape.
        bs (list of list of :class:`~chainer.Variable`): Bias vectors.
            ``bs[i + di]`` represnents biases for i-th layer.
            Note that ``di = 0`` for forward-RNN and ``di = 1`` for
            backward-RNN.
            Each ``bs[i + di]`` is a list containing two vectors.
            ``bs[i + di][j]`` is corresponding with ``b^{f}_j`` if ``di = 0``
            and corresponding with ``b^{b}_j`` if ``di = 1`` in the equation.
            Shape of each matrix is ``(N,)`` where ``N`` is dimension of
            hidden units.
        xs (list of :class:`~chainer.Variable`):
            A list of :class:`~chainer.Variable` holding input values.
            Each element ``xs[t]`` holds input value
            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is
            mini-batch size for time ``t``, and ``I`` is size of input units.
            Note that this function supports variable length sequences.
            When sequneces has different lengths, sort sequences in descending
            order by length, and transpose the sorted sequence.
            :func:`~chainer.functions.transpose_sequence` transpose a list
            of :func:`~chainer.Variable` holding sequence.
            So ``xs`` needs to satisfy
            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.
        activation (str): Activation function name.
            Please select ``tanh`` or ``relu``.

    Returns:
        tuple: This function returns a tuple containing three elements,
        ``hy`` and ``ys``.

        - ``hy`` is an updated hidden states whose shape is same as ``hx``.
        - ``ys`` is a list of :class:`~chainer.Variable` . Each element
          ``ys[t]`` holds hidden states of the last layer corresponding
          to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t``
          is mini-batch size for time ``t``, and ``N`` is size of hidden
          units. Note that ``B_t`` is the same value as ``xs[t]``.

    """
    return n_step_rnn_base(n_layers, dropout_ratio, hx, ws, bs, xs,
                           activation, use_bi_direction=True)


def n_step_rnn_base(n_layers, dropout_ratio, hx, ws, bs, xs,
                    activation, use_bi_direction, **kwargs):
    """n_step_rnn_base(n_layers, dropout_ratio, hx, ws, bs, xs, activation, \
use_bi_direction)

    Base function for Stack RNN/BiRNN functions.

    This function is used at  :func:`chainer.functions.n_step_birnn` and
    :func:`chainer.functions.n_step_rnn`.
    This function's behavior depends on following arguments,
    ``activation`` and ``use_bi_direction``.

    Args:
        n_layers(int): Number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (:class:`~chainer.Variable`):
            Variable holding stacked hidden states.
            Its shape is ``(S, B, N)`` where ``S`` is number of layers and is
            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
            dimension of hidden units.
        ws (list of list of :class:`~chainer.Variable`): Weight matrices.
            ``ws[i]`` represents weights for i-th layer.
            Each ``ws[i]`` is a list containing two matrices.
            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 1`` is ``(I, N)`` shape as they
            are multiplied with input variables. All other matrices has
            ``(N, N)`` shape.
        bs (list of list of :class:`~chainer.Variable`): Bias vectors.
            ``bs[i]`` represnents biases for i-th layer.
            Each ``bs[i]`` is a list containing two vectors.
            ``bs[i][j]`` is corresponding with ``b_j`` in the equation.
            Shape of each matrix is ``(N,)`` where ``N`` is dimension of
            hidden units.
        xs (list of :class:`~chainer.Variable`):
            A list of :class:`~chainer.Variable` holding input values.
            Each element ``xs[t]`` holds input value
            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is
            mini-batch size for time ``t``, and ``I`` is size of input units.
            Note that this function supports variable length sequences.
            When sequneces has different lengths, sort sequences in descending
            order by length, and transpose the sorted sequence.
            :func:`~chainer.functions.transpose_sequence` transpose a list
            of :func:`~chainer.Variable` holding sequence.
            So ``xs`` needs to satisfy
            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.
        activation (str): Activation function name.
            Please select ``tanh`` or ``relu``.
        use_bi_direction (bool): If ``True``, this function uses
            Bi-directional RNN.

    Returns:
        tuple: This function returns a tuple containing three elements,
            ``hy`` and ``ys``.

            - ``hy`` is an updated hidden states whose shape is same as ``hx``.
            - ``ys`` is a list of :class:`~chainer.Variable` . Each element
              ``ys[t]`` holds hidden states of the last layer corresponding
              to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t``
              is mini-batch size for time ``t``, and ``N`` is size of hidden
              units. Note that ``B_t`` is the same value as ``xs[t]``.

    .. seealso::
       :func:`chainer.functions.n_step_rnn`
       :func:`chainer.functions.n_step_birnn`

    """
    if kwargs:
        argument.check_unexpected_kwargs(
            kwargs, train='train argument is not supported anymore. '
            'Use chainer.using_config',
            use_cudnn='use_cudnn argument is not supported anymore. '
            'Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)

    activation_list = ['tanh', 'relu']
    if activation not in activation_list:
        candidate = ','.join(activation_list)
        raise ValueError('Invalid activation: "%s". Please select from [%s]'
                         % (activation, candidate))

    # Check input size consistency with xs and ws.
    x_in = xs[0].shape[1]
    w_in = ws[0][0].shape[1]
    if x_in != w_in:
        raise ValueError('Inconsistent input size in input values and weight '
                         'parameters: {} != {}'.format(x_in, w_in))

    xp = backend.get_array_module(hx)

    if xp is cuda.cupy and chainer.should_use_cudnn('>=auto', 5000):
        lengths = [len(x) for x in xs]
        xs = chainer.functions.concat(xs, axis=0)
        with chainer.using_device(xs.device):
            states = cuda.get_cudnn_dropout_states()
            states.set_dropout_ratio(dropout_ratio)

        rnn_mode = 'rnn_%s' % activation
        w = cudnn_rnn_weight_concat(
            n_layers, states, use_bi_direction, rnn_mode, ws, bs)

        if use_bi_direction:
            # Bi-directional RNN
            if activation == 'tanh':
                rnn = NStepBiRNNTanh
            elif activation == 'relu':
                rnn = NStepBiRNNReLU
        else:
            # Uni-directional RNN
            if activation == 'tanh':
                rnn = NStepRNNTanh
            elif activation == 'relu':
                rnn = NStepRNNReLU

        hy, ys = rnn(n_layers, states, lengths)(hx, w, xs)
        sections = numpy.cumsum(lengths[:-1])
        ys = chainer.functions.split_axis(ys, sections, 0)
        return hy, ys

    else:

        def f(x, h, c, w, b):
            xw, hw = w
            xb, hb = b
            rnn_in = linear.linear(x, xw, xb) + linear.linear(h, hw, hb)
            if activation == 'tanh':
                return tanh.tanh(rnn_in), None
            elif activation == 'relu':
                return relu.relu(rnn_in), None

        hy, _, ys = n_step_rnn_impl(
            f, n_layers, dropout_ratio, hx, None, ws, bs, xs, use_bi_direction)
        return hy, ys


def n_step_rnn_impl(
        f, n_layers, dropout_ratio, hx, cx, ws, bs, xs, use_bi_direction):
    direction = 2 if use_bi_direction else 1
    hx = chainer.functions.separate(hx)
    use_cell = cx is not None
    if use_cell:
        cx = chainer.functions.separate(cx)
    else:
        cx = [None] * len(hx)

    xs_next = xs
    hy = []
    cy = []
    for layer in six.moves.range(n_layers):

        # Forward RNN
        if layer == 0:
            xs = xs_next
        else:
            xs = _dropout_sequence(xs_next, dropout_ratio)
        idx = direction * layer
        h, c, h_forward = _one_directional_loop(
            f, xs, hx[idx], cx[idx], ws[idx], bs[idx])
        hy.append(h)
        cy.append(c)

        if use_bi_direction:
            # Backward RNN
            idx = direction * layer + 1
            if layer == 0:
                xs = xs_next
            else:
                xs = _dropout_sequence(xs_next, dropout_ratio)
            h, c, h_backward = _one_directional_loop(
                f, reversed(xs), hx[idx], cx[idx], ws[idx], bs[idx])
            h_backward.reverse()
            # Concat
            xs_next = [concat.concat([hfi, hbi], axis=1) for hfi, hbi in
                       six.moves.zip(h_forward, h_backward)]
            hy.append(h)
            cy.append(c)
        else:
            # Uni-directional RNN
            xs_next = h_forward

    ys = xs_next
    hy = stack.stack(hy)
    if use_cell:
        cy = stack.stack(cy)
    else:
        cy = None
    return hy, cy, tuple(ys)


def _one_directional_loop(f, xs, h, c, w, b):
    h_list = []
    for x in xs:
        batch = len(x)
        need_split = len(h) > batch
        if need_split:
            h, h_rest = split_axis.split_axis(h, [batch], axis=0)
            if c is not None:
                c, c_rest = split_axis.split_axis(c, [batch], axis=0)

        h, c = f(x, h, c, w, b)
        h_list.append(h)

        if need_split:
            h = concat.concat([h, h_rest], axis=0)
            if c is not None:
                c = concat.concat([c, c_rest], axis=0)
    return h, c, h_list


def _dropout_sequence(xs, dropout_ratio):
    return [dropout.dropout(x, ratio=dropout_ratio) for x in xs]
