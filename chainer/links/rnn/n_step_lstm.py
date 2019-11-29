from chainer.functions.rnn import n_step_lstm as rnn
from chainer import initializers
from chainer.links.rnn import n_step_rnn


class NStepLSTMBase(n_step_rnn.NStepRNNBase):
    """Base link class for Stacked LSTM/BiLSTM links.

    This link is base link class for :func:`chainer.links.NStepLSTM` and
    :func:`chainer.links.NStepBiLSTM`.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.
        lateral_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the lateral connections.
            May be ``None`` to use default initialization.
        upward_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the upward connections.
            May be ``None`` to use default initialization.
        bias_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value
            It is used for initialization of the biases of cell input,
            input gate and output gate.and gates of the upward connection.
            May be a scalar, in that case, the bias is
            initialized by this value.
            If it is ``None``, the cell-input bias is initialized to zero.
        forget_bias_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value
            It is used for initialization of the biases of the forget gate of
            the upward connection.
            May be a scalar, in that case, the bias is
            initialized by this value.
            If it is ``None``, the forget bias is initialized to one.

    .. note::

        In Chainer v7, the default value of `forget_bias_init` will be
        changed from zero to one to make it consistent with
        :class:`~chainer.links.LSTM`.

    .. seealso::
        :func:`chainer.functions.n_step_lstm`
        :func:`chainer.functions.n_step_bilstm`

    """

    # Update docstring of subclasses accordingly when modifying docstring.

    n_weights = 8

    def __init__(self, *args, **kwargs):
        lateral_init = kwargs.pop("lateral_init", None)
        upward_init = kwargs.pop("upward_init", None)
        bias_init = kwargs.pop("bias_init", None)
        forget_bias_init = kwargs.pop("forget_bias_init", 1)

        if (kwargs.get('initialW', None) is not None and
                (lateral_init is not None or upward_init is not None)):
            raise ValueError('initialW and lateral_init/upward_init '
                             'cannot be specified together')

        if (kwargs.get('initial_bias', None) is not None and
                (bias_init is not None or forget_bias_init is not None)):
            raise ValueError('initial_bias and bias_init/forget_bias_init '
                             'cannot be specified together')

        super(NStepLSTMBase, self).__init__(*args, **kwargs)

        if lateral_init is not None:
            lateral_init = initializers._get_initializer(lateral_init)
            for ws_i in self.ws:
                for w in (ws_i[4], ws_i[5], ws_i[6], ws_i[7]):
                    w.initializer = lateral_init

        if upward_init is not None:
            upward_init = initializers._get_initializer(upward_init)
            for ws_i in self.ws:
                for w in (ws_i[0], ws_i[1], ws_i[2], ws_i[3]):
                    w.initializer = upward_init

        if bias_init is not None:
            bias_init = initializers._get_initializer(bias_init)
            # Leave b{4,6,7} as zero to avoid doubling the effective bias
            # for each gate.
            for bs_i in self.bs:
                for b in (bs_i[0], bs_i[2], bs_i[3]):
                    b.initializer = bias_init

        if forget_bias_init is not None:
            forget_bias_init = initializers._get_initializer(forget_bias_init)
            # Leave b5 as zero to avoid doubling the effective bias
            # for each gate.
            for bs_i in self.bs:
                bs_i[1].initializer = forget_bias_init

    def forward(self, hx, cx, xs, **kwargs):
        """forward(self, hx, cx, xs)

        Calculates all of the hidden states and the cell states.

        Args:
            hx (:class:`~chainer.Variable` or None):
                Initial hidden states. If ``None`` is specified zero-vector
                is used. Its shape is ``(S, B, N)`` for uni-directional LSTM
                and ``(2S, B, N)`` for bi-directional LSTM where ``S`` is
                the number of layers and is equal to ``n_layers``,
                ``B`` is the mini-batch size,
                and ``N`` is the dimension of the hidden units.
            cx (:class:`~chainer.Variable` or None):
                Initial cell states. If ``None`` is specified zero-vector is
                used.  It has the same shape as ``hx``.
            xs (list of :class:`~chainer.Variable`): List of input sequences.
                Each element ``xs[i]`` is a :class:`chainer.Variable` holding
                a sequence. Its shape is ``(L_i, I)``, where ``L_i`` is the
                length of a sequence for batch ``i``, and ``I`` is the size of
                the input and is equal to ``in_size``.

        Returns:
            tuple: This function returns a tuple containing three elements,
            ``hy``, ``cy`` and ``ys``.

            - ``hy`` is an updated hidden states whose shape is the same as
              ``hx``.
            - ``cy`` is an updated cell states whose shape is the same as
              ``cx``.
            - ``ys`` is a list of :class:`~chainer.Variable` . Each element
              ``ys[i]`` holds hidden states of the last layer corresponding
              to an input ``xs[i]``. Its shape is ``(L_i, N)`` for
              uni-directional LSTM and ``(L_i, 2N)`` for bi-directional LSTM
              where ``L_i`` is the length of a sequence for batch ``i``,
              and ``N`` is size of hidden units.
        """
        (hy, cy), ys = self._call([hx, cx], xs, **kwargs)
        return hy, cy, ys


class NStepLSTM(NStepLSTMBase):
    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Uni-directional LSTM for sequences.

    This link is stacked version of Uni-directional LSTM for sequences.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_lstm`, this function automatically
    sort inputs in descending order by length, and transpose the sequence.
    Users just need to call the link with a list of :class:`chainer.Variable`
    holding sequences.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.
        lateral_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the lateral connections.
            May be ``None`` to use default initialization.
        upward_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the upward connections.
            May be ``None`` to use default initialization.
        bias_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value
            It is used for initialization of the biases of cell input,
            input gate and output gate.and gates of the upward connection.
            May be a scalar, in that case, the bias is
            initialized by this value.
            If it is ``None``, the cell-input bias is initialized to zero.
        forget_bias_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value
            It is used for initialization of the biases of the forget gate of
            the upward connection.
            May be a scalar, in that case, the bias is
            initialized by this value.
            If it is ``None``, the forget bias is initialized to one.

    .. note::

        In Chainer v7, the default value of `forget_bias_init` will be
        changed from zero to one to make it consistent with
        :class:`~chainer.links.LSTM`.

    .. seealso::
        :func:`chainer.functions.n_step_lstm`

    .. admonition:: Example

        *Read* :meth:`forward` *method below first.*

        >>> dropout_ratio = 0.0
        >>> in_size, seq_len, n_layers, out_size = 2, 4, 2, 3
        >>> batch = 5
        >>> xs = [
        ...     Variable(np.random.rand(seq_len, in_size).astype(np.float32))
        ...     for i in range(batch)]
        >>> [x.shape for x in xs]
        [(4, 2), (4, 2), (4, 2), (4, 2), (4, 2)]
        >>> lstm = L.NStepLSTM(n_layers, in_size, out_size, dropout_ratio)

        Without hidden or cell state:

        >>> hy, cy, ys = lstm(None, None, xs)
        >>> hy.shape  # shape should be (n_layers, batch, out_size)
        (2, 5, 3)
        >>> ys[0].shape  # should be (seq_len, out_size)
        (4, 3)
        >>> len(ys)  # should be equal to batch
        5

        With hidden and cell states:

        >>> h_shape = (n_layers, batch, out_size)
        >>> hx = Variable(np.ones(h_shape, np.float32))
        >>> cx = Variable(np.ones(h_shape, np.float32))
        >>> hy, cy, ys = lstm(hx, cx, xs)
        >>> hy.shape  # shape should be (n_layers, batch, out_size)
        (2, 5, 3)
        >>> ys[0].shape  # should be (seq_len, out_size)
        (4, 3)

    """

    use_bi_direction = False

    def rnn(self, *args):
        return rnn.n_step_lstm(*args)

    @property
    def n_cells(self):
        return 2


class NStepBiLSTM(NStepLSTMBase):
    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Bi-directional LSTM for sequences.

    This link is stacked version of Bi-directional LSTM for sequences.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_bilstm`, this function automatically
    sort inputs in descending order by length, and transpose the sequence.
    Users just need to call the link with a list of :class:`chainer.Variable`
    holding sequences.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.
        lateral_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the lateral connections.
            May be ``None`` to use default initialization.
        upward_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the upward connections.
            May be ``None`` to use default initialization.
        bias_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value
            It is used for initialization of the biases of cell input,
            input gate and output gate.and gates of the upward connection.
            May be a scalar, in that case, the bias is
            initialized by this value.
            If it is ``None``, the cell-input bias is initialized to zero.
        forget_bias_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value
            It is used for initialization of the biases of the forget gate of
            the upward connection.
            May be a scalar, in that case, the bias is
            initialized by this value.
            If it is ``None``, the forget bias is initialized to one.

    .. note::

        In Chainer v7, the default value of `forget_bias_init` will be
        changed from zero to one to make it consistent with
        :class:`~chainer.links.LSTM`.

    .. seealso::
        :func:`chainer.functions.n_step_bilstm`

    """

    use_bi_direction = True

    def rnn(self, *args):
        return rnn.n_step_bilstm(*args)

    @property
    def n_cells(self):
        return 2
