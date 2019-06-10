import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer.functions.connection import linear
from chainer.functions.connection import n_step_rnn
from chainer.utils import argument


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn


class NStepGRU(n_step_rnn.BaseNStepRNN):

    def __init__(self, n_layers, states, lengths, **kwargs):
        n_step_rnn.BaseNStepRNN.__init__(
            self, n_layers, states, lengths,
            rnn_dir='uni', rnn_mode='gru', **kwargs)


class NStepBiGRU(n_step_rnn.BaseNStepRNN):

    def __init__(self, n_layers, states, lengths, **kwargs):
        n_step_rnn.BaseNStepRNN.__init__(
            self, n_layers, states, lengths,
            rnn_dir='bi', rnn_mode='gru', **kwargs)


def n_step_gru(
        n_layers, dropout_ratio, hx, ws, bs, xs, **kwargs):
    """n_step_gru(n_layers, dropout_ratio, hx, ws, bs, xs)

    Stacked Uni-directional Gated Recurrent Unit function.

    This function calculates stacked Uni-directional GRU with sequences.
    This function gets an initial hidden state :math:`h_0`, an input
    sequence :math:`x`, weight matrices :math:`W`, and bias vectors :math:`b`.
    This function calculates hidden states :math:`h_t` for each time :math:`t`
    from input :math:`x_t`.

    .. math::
       r_t &= \\sigma(W_0 x_t + W_3 h_{t-1} + b_0 + b_3) \\\\
       z_t &= \\sigma(W_1 x_t + W_4 h_{t-1} + b_1 + b_4) \\\\
       h'_t &= \\tanh(W_2 x_t + b_2 + r_t \\cdot (W_5 h_{t-1} + b_5)) \\\\
       h_t &= (1 - z_t) \\cdot h'_t + z_t \\cdot h_{t-1}

    As the function accepts a sequence, it calculates :math:`h_t` for all
    :math:`t` with one call. Six weight matrices and six bias vectors are
    required for each layers. So, when :math:`S` layers exists, you need to
    prepare :math:`6S` weight matrices and :math:`6S` bias vectors.

    If the number of layers ``n_layers`` is greather than :math:`1`, input
    of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.
    Note that all input variables except first layer may have different shape
    from the first layer.

    Args:
        n_layers(int): Number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (~chainer.Variable):
            Variable holding stacked hidden states.
            Its shape is ``(S, B, N)`` where ``S`` is number of layers and is
            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
            dimension of hidden units.
        ws (list of list of :class:`~chainer.Variable`): Weight matrices.
            ``ws[i]`` represents weights for i-th layer.
            Each ``ws[i]`` is a list containing six matrices.
            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 3`` is ``(I, N)`` shape as they
            are multiplied with input variables. All other matrices has
            ``(N, N)`` shape.
        bs (list of list of :class:`~chainer.Variable`): Bias vectors.
            ``bs[i]`` represnents biases for i-th layer.
            Each ``bs[i]`` is a list containing six vectors.
            ``bs[i][j]`` is corresponding with ``b_j`` in the equation.
            Shape of each matrix is ``(N,)`` where ``N`` is dimension of
            hidden units.
        xs (list of :class:`~chainer.Variable`):
            A list of :class:`~chainer.Variable`
            holding input values. Each element ``xs[t]`` holds input value
            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is
            mini-batch size for time ``t``, and ``I`` is size of input units.
            Note that this function supports variable length sequences.
            When sequneces has different lengths, sort sequences in descending
            order by length, and transpose the sorted sequence.
            :func:`~chainer.functions.transpose_sequence` transpose a list
            of :func:`~chainer.Variable` holding sequence.
            So ``xs`` needs to satisfy
            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.

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

    return n_step_gru_base(n_layers, dropout_ratio, hx, ws, bs, xs,
                           use_bi_direction=False, **kwargs)


def n_step_bigru(
        n_layers, dropout_ratio, hx, ws, bs, xs, **kwargs):
    """n_step_bigru(n_layers, dropout_ratio, hx, ws, bs, xs)

    Stacked Bi-directional Gated Recurrent Unit function.

    This function calculates stacked Bi-directional GRU with sequences.
    This function gets an initial hidden state :math:`h_0`, an input
    sequence :math:`x`, weight matrices :math:`W`, and bias vectors :math:`b`.
    This function calculates hidden states :math:`h_t` for each time :math:`t`
    from input :math:`x_t`.

    .. math::
       r^{f}_t &= \\sigma(W^{f}_0 x_t + W^{f}_3 h_{t-1} + b^{f}_0 + b^{f}_3)
       \\\\
       z^{f}_t &= \\sigma(W^{f}_1 x_t + W^{f}_4 h_{t-1} + b^{f}_1 + b^{f}_4)
       \\\\
       h^{f'}_t &= \\tanh(W^{f}_2 x_t + b^{f}_2 + r^{f}_t \\cdot (W^{f}_5
       h_{t-1} + b^{f}_5)) \\\\
       h^{f}_t &= (1 - z^{f}_t) \\cdot h^{f'}_t + z^{f}_t \\cdot h_{t-1}
       \\\\
       r^{b}_t &= \\sigma(W^{b}_0 x_t + W^{b}_3 h_{t-1} + b^{b}_0 + b^{b}_3)
       \\\\
       z^{b}_t &= \\sigma(W^{b}_1 x_t + W^{b}_4 h_{t-1} + b^{b}_1 + b^{b}_4)
       \\\\
       h^{b'}_t &= \\tanh(W^{b}_2 x_t + b^{b}_2 + r^{b}_t \\cdot (W^{b}_5
       h_{t-1} + b^{b}_5)) \\\\
       h^{b}_t &= (1 - z^{b}_t) \\cdot h^{b'}_t + z^{b}_t \\cdot h_{t-1}
       \\\\
       h_t  &= [h^{f}_t; h^{b}_t] \\\\

    where :math:`W^{f}` is weight matrices for forward-GRU, :math:`W^{b}` is
    weight matrices for backward-GRU.

    As the function accepts a sequence, it calculates :math:`h_t` for all
    :math:`t` with one call. Six weight matrices and six bias vectors are
    required for each layers. So, when :math:`S` layers exists, you need to
    prepare :math:`6S` weight matrices and :math:`6S` bias vectors.

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
            dimension of hidden units.
        ws (list of list of :class:`~chainer.Variable`): Weight matrices.
            ``ws[i]`` represents weights for i-th layer.
            Each ``ws[i]`` is a list containing six matrices.
            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 3`` is ``(I, N)`` shape as they
            are multiplied with input variables. All other matrices has
            ``(N, N)`` shape.
        bs (list of list of :class:`~chainer.Variable`): Bias vectors.
            ``bs[i]`` represnents biases for i-th layer.
            Each ``bs[i]`` is a list containing six vectors.
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
        use_bi_direction (bool): If ``True``, this function uses
            Bi-direction GRU.

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

    return n_step_gru_base(n_layers, dropout_ratio, hx, ws, bs, xs,
                           use_bi_direction=True, **kwargs)


def n_step_gru_base(n_layers, dropout_ratio, hx, ws, bs, xs,
                    use_bi_direction, **kwargs):
    """n_step_gru_base(n_layers, dropout_ratio, hx, ws, bs, xs, \
use_bi_direction)

    Base function for Stack GRU/BiGRU functions.

    This function is used at  :func:`chainer.functions.n_step_bigru` and
    :func:`chainer.functions.n_step_gru`.
    This function's behavior depends on argument ``use_bi_direction``.

    Args:
        n_layers(int): Number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (:class:`~chainer.Variable`):
            Variable holding stacked hidden states.
            Its shape is ``(S, B, N)`` where ``S`` is number of layers and is
            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
            dimension of hidden units. Because of bi-direction, the
            first dimension length is ``2S``.
        ws (list of list of :class:`~chainer.Variable`): Weight matrices.
            ``ws[i]`` represents weights for i-th layer.
            Each ``ws[i]`` is a list containing six matrices.
            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 3`` is ``(I, N)`` shape as they
            are multiplied with input variables. All other matrices has
            ``(N, N)`` shape.
        bs (list of list of :class:`~chainer.Variable`): Bias vectors.
            ``bs[i]`` represnents biases for i-th layer.
            Each ``bs[i]`` is a list containing six vectors.
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
            Bi-direction GRU.

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

    xp = backend.get_array_module(hx, hx.data)

    if xp is cuda.cupy and chainer.should_use_cudnn('>=auto', 5000):
        lengths = [len(x) for x in xs]
        xs = chainer.functions.concat(xs, axis=0)
        with chainer.using_device(xs.device):
            states = cuda.get_cudnn_dropout_states()
            states.set_dropout_ratio(dropout_ratio)

        w = n_step_rnn.cudnn_rnn_weight_concat(
            n_layers, states, use_bi_direction, 'gru', ws, bs)

        if use_bi_direction:
            rnn = NStepBiGRU
        else:
            rnn = NStepGRU

        hy, ys = rnn(n_layers, states, lengths)(hx, w, xs)
        sections = numpy.cumsum(lengths[:-1])
        ys = chainer.functions.split_axis(ys, sections, 0)
        return hy, ys

    else:
        hy, _, ys = n_step_rnn.n_step_rnn_impl(
            _gru, n_layers, dropout_ratio, hx, None, ws, bs, xs,
            use_bi_direction)
        return hy, ys


def _gru(x, h, c, w, b):
    xw = concat.concat([w[0], w[1], w[2]], axis=0)
    hw = concat.concat([w[3], w[4], w[5]], axis=0)
    xb = concat.concat([b[0], b[1], b[2]], axis=0)
    hb = concat.concat([b[3], b[4], b[5]], axis=0)

    gru_x = linear.linear(x, xw, xb)
    gru_h = linear.linear(h, hw, hb)

    W_r_x, W_z_x, W_x = split_axis.split_axis(gru_x, 3, axis=1)
    U_r_h, U_z_h, U_x = split_axis.split_axis(gru_h, 3, axis=1)

    r = sigmoid.sigmoid(W_r_x + U_r_h)
    z = sigmoid.sigmoid(W_z_x + U_z_h)
    h_bar = tanh.tanh(W_x + r * U_x)
    return (1 - z) * h_bar + z * h, None
