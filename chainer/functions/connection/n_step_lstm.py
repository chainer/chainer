import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.functions.activation import lstm
from chainer.functions.array import reshape
from chainer.functions.array import stack
from chainer.functions.connection import linear
from chainer.functions.connection import n_step_rnn
from chainer.utils import argument
import chainerx


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn


def _stack_weight(ws):
    # TODO(unno): Input of the current LSTM implementation is shuffled
    w = stack.stack(ws, axis=1)
    shape = w.shape
    return reshape.reshape(w, (shape[0] * shape[1],) + shape[2:])


class NStepLSTM(n_step_rnn.BaseNStepRNN):

    def __init__(self, n_layers, states, lengths):
        n_step_rnn.BaseNStepRNN.__init__(
            self, n_layers, states, lengths,
            rnn_dir='uni', rnn_mode='lstm')


class NStepBiLSTM(n_step_rnn.BaseNStepRNN):

    def __init__(self, n_layers, states, lengths):
        n_step_rnn.BaseNStepRNN.__init__(
            self, n_layers, states, lengths,
            rnn_dir='bi', rnn_mode='lstm')


def n_step_lstm(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs, **kwargs):
    """n_step_lstm(n_layers, dropout_ratio, hx, cx, ws, bs, xs)

    Stacked Uni-directional Long Short-Term Memory function.

    This function calculates stacked Uni-directional LSTM with sequences.
    This function gets an initial hidden state :math:`h_0`, an initial cell
    state :math:`c_0`, an input sequence :math:`x`, weight matrices :math:`W`,
    and bias vectors :math:`b`.
    This function calculates hidden states :math:`h_t` and :math:`c_t` for each
    time :math:`t` from input :math:`x_t`.

    .. math::
       i_t &= \\sigma(W_0 x_t + W_4 h_{t-1} + b_0 + b_4) \\\\
       f_t &= \\sigma(W_1 x_t + W_5 h_{t-1} + b_1 + b_5) \\\\
       o_t &= \\sigma(W_2 x_t + W_6 h_{t-1} + b_2 + b_6) \\\\
       a_t &= \\tanh(W_3 x_t + W_7 h_{t-1} + b_3 + b_7) \\\\
       c_t &= f_t \\cdot c_{t-1} + i_t \\cdot a_t \\\\
       h_t &= o_t \\cdot \\tanh(c_t)

    As the function accepts a sequence, it calculates :math:`h_t` for all
    :math:`t` with one call. Eight weight matrices and eight bias vectors are
    required for each layer. So, when :math:`S` layers exist, you need to
    prepare :math:`8S` weight matrices and :math:`8S` bias vectors.

    If the number of layers ``n_layers`` is greater than :math:`1`, the input
    of the ``k``-th layer is the hidden state ``h_t`` of the ``k-1``-th layer.
    Note that all input variables except the first layer may have different
    shape from the first layer.

    Args:
        n_layers(int): The number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (:class:`~chainer.Variable`):
            Variable holding stacked hidden states.
            Its shape is ``(S, B, N)`` where ``S`` is the number of layers and
            is equal to ``n_layers``, ``B`` is the mini-batch size, and ``N``
            is the dimension of the hidden units.
        cx (:class:`~chainer.Variable`): Variable holding stacked cell states.
            It has the same shape as ``hx``.
        ws (list of list of :class:`~chainer.Variable`): Weight matrices.
            ``ws[i]`` represents the weights for the i-th layer.
            Each ``ws[i]`` is a list containing eight matrices.
            ``ws[i][j]`` corresponds to :math:`W_j` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 4`` are ``(I, N)``-shaped as
            they are multiplied with input variables, where ``I`` is the size
            of the input and ``N`` is the dimension of the hidden units. All
            other matrices are ``(N, N)``-shaped.
        bs (list of list of :class:`~chainer.Variable`): Bias vectors.
            ``bs[i]`` represents the biases for the i-th layer.
            Each ``bs[i]`` is a list containing eight vectors.
            ``bs[i][j]`` corresponds to :math:`b_j` in the equation.
            The shape of each matrix is ``(N,)`` where ``N`` is the dimension
            of the hidden units.
        xs (list of :class:`~chainer.Variable`):
            A list of :class:`~chainer.Variable`
            holding input values. Each element ``xs[t]`` holds input value
            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is the
            mini-batch size for time ``t``. The sequences must be transposed.
            :func:`~chainer.functions.transpose_sequence` can be used to
            transpose a list of :class:`~chainer.Variable`\\ s each
            representing a sequence.
            When sequences has different lengths, they must be
            sorted in descending order of their lengths before transposing.
            So ``xs`` needs to satisfy
            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.

    Returns:
        tuple: This function returns a tuple containing three elements,
        ``hy``, ``cy`` and ``ys``.

        - ``hy`` is an updated hidden states whose shape is the same as
          ``hx``.
        - ``cy`` is an updated cell states whose shape is the same as
          ``cx``.
        - ``ys`` is a list of :class:`~chainer.Variable` . Each element
          ``ys[t]`` holds hidden states of the last layer corresponding
          to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is
          the mini-batch size for time ``t``, and ``N`` is size of hidden
          units. Note that ``B_t`` is the same value as ``xs[t]``.

    .. note::

       The dimension of hidden units is limited to only one size ``N``. If you
       want to use variable dimension of hidden units, please use
       :class:`chainer.functions.lstm`.

    .. seealso::

       :func:`chainer.functions.lstm`

    .. admonition:: Example

        >>> batchs = [3, 2, 1]  # support variable length sequences
        >>> in_size, out_size, n_layers = 3, 2, 2
        >>> dropout_ratio = 0.0
        >>> xs = [np.ones((b, in_size)).astype(np.float32) for b in batchs]
        >>> [x.shape for x in xs]
        [(3, 3), (2, 3), (1, 3)]
        >>> h_shape = (n_layers, batchs[0], out_size)
        >>> hx = np.ones(h_shape).astype(np.float32)
        >>> cx = np.ones(h_shape).astype(np.float32)
        >>> w_in = lambda i, j: in_size if i == 0 and j < 4 else out_size
        >>> ws = []
        >>> bs = []
        >>> for n in range(n_layers):
        ...     ws.append([np.ones((out_size, w_in(n, i))).astype(np.float32) \
for i in range(8)])
        ...     bs.append([np.ones((out_size,)).astype(np.float32) \
for _ in range(8)])
        ...
        >>> ws[0][0].shape  # ws[0][:4].shape are (out_size, in_size)
        (2, 3)
        >>> ws[1][0].shape  # others are (out_size, out_size)
        (2, 2)
        >>> bs[0][0].shape
        (2,)
        >>> hy, cy, ys = F.n_step_lstm(
        ...     n_layers, dropout_ratio, hx, cx, ws, bs, xs)
        >>> hy.shape
        (2, 3, 2)
        >>> cy.shape
        (2, 3, 2)
        >>> [y.shape for y in ys]
        [(3, 2), (2, 2), (1, 2)]

    """

    return n_step_lstm_base(n_layers, dropout_ratio, hx, cx, ws, bs, xs,
                            use_bi_direction=False, **kwargs)


def n_step_bilstm(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs, **kwargs):
    """n_step_bilstm(n_layers, dropout_ratio, hx, cx, ws, bs, xs)

    Stacked Bi-directional Long Short-Term Memory function.

    This function calculates stacked Bi-directional LSTM with sequences.
    This function gets an initial hidden state :math:`h_0`, an initial cell
    state :math:`c_0`, an input sequence :math:`x`, weight matrices :math:`W`,
    and bias vectors :math:`b`.
    This function calculates hidden states :math:`h_t` and :math:`c_t` for each
    time :math:`t` from input :math:`x_t`.

    .. math::
        i^{f}_t &=& \\sigma(W^{f}_0 x_t + W^{f}_4 h_{t-1} + b^{f}_0 + b^{f}_4),
        \\\\
        f^{f}_t &=& \\sigma(W^{f}_1 x_t + W^{f}_5 h_{t-1} + b^{f}_1 + b^{f}_5),
        \\\\
        o^{f}_t &=& \\sigma(W^{f}_2 x_t + W^{f}_6 h_{t-1} + b^{f}_2 + b^{f}_6),
        \\\\
        a^{f}_t &=& \\tanh(W^{f}_3 x_t + W^{f}_7 h_{t-1} + b^{f}_3 + b^{f}_7),
        \\\\
        c^{f}_t &=& f^{f}_t \\cdot c^{f}_{t-1} + i^{f}_t \\cdot a^{f}_t,
        \\\\
        h^{f}_t &=& o^{f}_t \\cdot \\tanh(c^{f}_t),
        \\\\
        i^{b}_t &=& \\sigma(W^{b}_0 x_t + W^{b}_4 h_{t-1} + b^{b}_0 + b^{b}_4),
        \\\\
        f^{b}_t &=& \\sigma(W^{b}_1 x_t + W^{b}_5 h_{t-1} + b^{b}_1 + b^{b}_5),
        \\\\
        o^{b}_t &=& \\sigma(W^{b}_2 x_t + W^{b}_6 h_{t-1} + b^{b}_2 + b^{b}_6),
        \\\\
        a^{b}_t &=& \\tanh(W^{b}_3 x_t + W^{b}_7 h_{t-1} + b^{b}_3 + b^{b}_7),
        \\\\
        c^{b}_t &=& f^{b}_t \\cdot c^{b}_{t-1} + i^{b}_t \\cdot a^{b}_t, \\\\
        h^{b}_t &=& o^{b}_t \\cdot \\tanh(c^{b}_t), \\\\
        h_t &=& [h^{f}_t; h^{b}_t]

    where :math:`W^{f}` is the weight matrices for forward-LSTM, :math:`W^{b}`
    is weight matrices for backward-LSTM.

    As the function accepts a sequence, it calculates :math:`h_t` for all
    :math:`t` with one call. Eight weight matrices and eight bias vectors are
    required for each layer of each direction. So, when :math:`S` layers
    exist, you need to prepare :math:`16S` weight matrices and :math:`16S`
    bias vectors.

    If the number of layers ``n_layers`` is greater than :math:`1`, the input
    of the ``k``-th layer is the hidden state ``h_t`` of the ``k-1``-th layer.
    Note that all input variables except the first layer may have different
    shape from the first layer.

    Args:
        n_layers(int): The number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (:class:`~chainer.Variable`):
            Variable holding stacked hidden states.
            Its shape is ``(2S, B, N)`` where ``S`` is the number of layers and
            is equal to ``n_layers``, ``B`` is the mini-batch size, and ``N``
            is the dimension of the hidden units. Because of bi-direction, the
            first dimension length is ``2S``.
        cx (:class:`~chainer.Variable`): Variable holding stacked cell states.
            It has the same shape as ``hx``.
        ws (list of list of :class:`~chainer.Variable`): Weight matrices.
            ``ws[2 * l + m]`` represents the weights for the l-th layer of
            the m-th direction. (``m == 0`` means the forward direction and
            ``m == 1`` means the backward direction.) Each ``ws[i]`` is a
            list containing eight matrices. ``ws[i][j]`` corresponds to
            :math:`W_j` in the equation. ``ws[0][j]`` and ``ws[1][j]`` where
            ``0 <= j < 4`` are ``(I, N)``-shaped because they are multiplied
            with input variables, where ``I`` is the size of the input.
            ``ws[i][j]`` where ``2 <= i`` and ``0 <= j < 4`` are
            ``(N, 2N)``-shaped because they are multiplied with two hidden
            layers :math:`h_t = [h^{f}_t; h^{b}_t]`. All other matrices are
            ``(N, N)``-shaped.
        bs (list of list of :class:`~chainer.Variable`): Bias vectors.
            ``bs[2 * l + m]`` represents the weights for the l-th layer of
            m-th direction. (``m == 0`` means the forward direction and
            ``m == 1`` means the backward direction.)
            Each ``bs[i]`` is a list containing eight vectors.
            ``bs[i][j]`` corresponds to :math:`b_j` in the equation.
            The shape of each matrix is ``(N,)``.
        xs (list of :class:`~chainer.Variable`):
            A list of :class:`~chainer.Variable`
            holding input values. Each element ``xs[t]`` holds input value
            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is the
            mini-batch size for time ``t``. The sequences must be transposed.
            :func:`~chainer.functions.transpose_sequence` can be used to
            transpose a list of :class:`~chainer.Variable`\\ s each
            representing a sequence.
            When sequences has different lengths, they must be
            sorted in descending order of their lengths before transposing.
            So ``xs`` needs to satisfy
            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.

    Returns:
        tuple: This function returns a tuple containing three elements,
        ``hy``, ``cy`` and ``ys``.

        - ``hy`` is an updated hidden states whose shape is the same as
          ``hx``.
        - ``cy`` is an updated cell states whose shape is the same as
          ``cx``.
        - ``ys`` is a list of :class:`~chainer.Variable` . Each element
          ``ys[t]`` holds hidden states of the last layer corresponding
          to an input ``xs[t]``. Its shape is ``(B_t, 2N)`` where ``B_t``
          is the mini-batch size for time ``t``, and ``N`` is size of
          hidden units. Note that ``B_t`` is the same value as ``xs[t]``.

    .. admonition:: Example

        >>> batchs = [3, 2, 1]  # support variable length sequences
        >>> in_size, out_size, n_layers = 3, 2, 2
        >>> dropout_ratio = 0.0
        >>> xs = [np.ones((b, in_size)).astype(np.float32) for b in batchs]
        >>> [x.shape for x in xs]
        [(3, 3), (2, 3), (1, 3)]
        >>> h_shape = (n_layers * 2, batchs[0], out_size)
        >>> hx = np.ones(h_shape).astype(np.float32)
        >>> cx = np.ones(h_shape).astype(np.float32)
        >>> def w_in(i, j):
        ...     if i == 0 and j < 4:
        ...         return in_size
        ...     elif i > 0 and j < 4:
        ...         return out_size * 2
        ...     else:
        ...         return out_size
        ...
        >>> ws = []
        >>> bs = []
        >>> for n in range(n_layers):
        ...     for direction in (0, 1):
        ...         ws.append([np.ones((out_size, w_in(n, i))).\
astype(np.float32) for i in range(8)])
        ...         bs.append([np.ones((out_size,)).astype(np.float32) \
for _ in range(8)])
        ...
        >>> ws[0][0].shape  # ws[0:2][:4].shape are (out_size, in_size)
        (2, 3)
        >>> ws[2][0].shape  # ws[2:][:4].shape are (out_size, 2 * out_size)
        (2, 4)
        >>> ws[0][4].shape  # others are (out_size, out_size)
        (2, 2)
        >>> bs[0][0].shape
        (2,)
        >>> hy, cy, ys = F.n_step_bilstm(
        ...     n_layers, dropout_ratio, hx, cx, ws, bs, xs)
        >>> hy.shape
        (4, 3, 2)
        >>> cy.shape
        (4, 3, 2)
        >>> [y.shape for y in ys]
        [(3, 4), (2, 4), (1, 4)]

    """
    return n_step_lstm_base(n_layers, dropout_ratio, hx, cx, ws, bs, xs,
                            use_bi_direction=True, **kwargs)


def n_step_lstm_base(
        n_layers, dropout_ratio, hx, cx, ws, bs, xs, use_bi_direction,
        **kwargs):
    """Base function for Stack LSTM/BiLSTM functions.

    This function is used at :func:`chainer.functions.n_step_lstm` and
    :func:`chainer.functions.n_step_bilstm`.
    This function's behavior depends on following arguments,
    ``activation`` and ``use_bi_direction``.

    Args:
        n_layers(int): The number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (:class:`~chainer.Variable`):
            Variable holding stacked hidden states.
            Its shape is ``(S, B, N)`` where ``S`` is the number of layers and
            is equal to ``n_layers``, ``B`` is the mini-batch size, and ``N``
            is the dimension of the hidden units.
        cx (:class:`~chainer.Variable`): Variable holding stacked cell states.
            It has the same shape as ``hx``.
        ws (list of list of :class:`~chainer.Variable`): Weight matrices.
            ``ws[i]`` represents the weights for the i-th layer.
            Each ``ws[i]`` is a list containing eight matrices.
            ``ws[i][j]`` corresponds to :math:`W_j` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 4`` are ``(I, N)``-shape as they
            are multiplied with input variables, where ``I`` is the size of
            the input and ``N`` is the dimension of the hidden units. All
            other matrices are ``(N, N)``-shaped.
        bs (list of list of :class:`~chainer.Variable`): Bias vectors.
            ``bs[i]`` represents the biases for the i-th layer.
            Each ``bs[i]`` is a list containing eight vectors.
            ``bs[i][j]`` corresponds to :math:`b_j` in the equation.
            The shape of each matrix is ``(N,)``.
        xs (list of :class:`~chainer.Variable`):
            A list of :class:`~chainer.Variable`
            holding input values. Each element ``xs[t]`` holds input value
            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is the
            mini-batch size for time ``t``. The sequences must be transposed.
            :func:`~chainer.functions.transpose_sequence` can be used to
            transpose a list of :class:`~chainer.Variable`\\ s each
            representing a sequence.
            When sequences has different lengths, they must be
            sorted in descending order of their lengths before transposing.
            So ``xs`` needs to satisfy
            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.
        use_bi_direction (bool): If ``True``, this function uses Bi-directional
            LSTM.

    Returns:
        tuple: This function returns a tuple containing three elements,
        ``hy``, ``cy`` and ``ys``.

            - ``hy`` is an updated hidden states whose shape is the same as
              ``hx``.
            - ``cy`` is an updated cell states whose shape is the same as
              ``cx``.
            - ``ys`` is a list of :class:`~chainer.Variable` . Each element
              ``ys[t]`` holds hidden states of the last layer corresponding
              to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is
              the mini-batch size for time ``t``. Note that ``B_t`` is the same
              value as ``xs[t]``.

    .. seealso::

       :func:`chainer.functions.n_step_lstm`
       :func:`chainer.functions.n_step_bilstm`

    """
    if kwargs:
        argument.check_unexpected_kwargs(
            kwargs, train='train argument is not supported anymore. '
            'Use chainer.using_config',
            use_cudnn='use_cudnn argument is not supported anymore. '
            'Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)

    # Check input size consistency with xs and ws here.
    x_in = xs[0].shape[1]
    w_in = ws[0][0].shape[1]
    if x_in != w_in:
        raise ValueError('Inconsistent input size in input values and weight '
                         'parameters: {} != {}'.format(x_in, w_in))

    xp = backend.get_array_module(hx, hx.data)

    # TODO(imanishi): Support ChainerX n_step_rnn
    use_cuda = xp is cuda.cupy or (
        xp is chainerx and hx.device.device.backend.name == 'cuda')

    if use_cuda and chainer.should_use_cudnn('>=auto', 5000):
        lengths = [len(x) for x in xs]
        xs = chainer.functions.concat(xs, axis=0)
        with chainer.using_device(xs.device):
            states = cuda.get_cudnn_dropout_states()
            states.set_dropout_ratio(dropout_ratio)

        w = n_step_rnn.cudnn_rnn_weight_concat(
            n_layers, states, use_bi_direction, 'lstm', ws, bs)

        if use_bi_direction:
            rnn = NStepBiLSTM
        else:
            rnn = NStepLSTM

        hy, cy, ys = rnn(n_layers, states, lengths)(hx, cx, w, xs)
        sections = numpy.cumsum(lengths[:-1])
        ys = chainer.functions.split_axis(ys, sections, 0)
        return hy, cy, ys

    else:
        return n_step_rnn.n_step_rnn_impl(
            _lstm, n_layers, dropout_ratio, hx, cx, ws, bs, xs,
            use_bi_direction)


def _lstm(x, h, c, w, b):
    xw = _stack_weight([w[2], w[0], w[1], w[3]])
    hw = _stack_weight([w[6], w[4], w[5], w[7]])
    xb = _stack_weight([b[2], b[0], b[1], b[3]])
    hb = _stack_weight([b[6], b[4], b[5], b[7]])
    lstm_in = linear.linear(x, xw, xb) + linear.linear(h, hw, hb)
    c_bar, h_bar = lstm.lstm(c, lstm_in)
    return h_bar, c_bar
