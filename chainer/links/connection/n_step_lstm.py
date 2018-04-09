from chainer.functions.connection import n_step_lstm as rnn
from chainer.links.connection import n_step_rnn


class NStepLSTMBase(n_step_rnn.NStepRNNBase):
    """Base link class for Stacked LSTM/BiLSTM links.

    This link is base link class for :func:`chainer.links.NStepLSTM` and
    :func:`chainer.links.NStepBiLSTM`.

    This link's behavior depends on argument, ``use_bi_direction``.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.
        initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 2.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should be 1.
        use_bi_direction (bool): if ``True``, use Bi-directional LSTM.

    .. seealso::
        :func:`chainer.functions.n_step_lstm`
        :func:`chainer.functions.n_step_bilstm`

    """

    n_weights = 8

    def __call__(self, hx, cx, xs, **kwargs):
        """__call__(self, hx, cx, xs)

        Calculate all hidden states and cell states.

        .. warning::

           ``train`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', train)``.
           See :func:`chainer.using_config`.

        Args:
            hx (~chainer.Variable or None): Initial hidden states. If ``None``
                is specified zero-vector is used. Its shape is ``(S, B, N)``
                for uni-directional LSTM and ``(2S, B, N)`` for
                bi-directional LSTM where ``S`` is the number of layers
                and is equal to ``n_layers``, ``B`` is the mini-batch size,
                and ``N`` is the dimension of the hidden units.
            cx (~chainer.Variable or None): Initial cell states. If ``None``
                is specified zero-vector is used.
                It has the same shape as ``hx``.
            xs (list of ~chainer.Variable): List of input sequences.
                Each element ``xs[i]`` is a :class:`chainer.Variable` holding
                a sequence. Its shape is ``(L_t, I)``, where ``L_t`` is the
                length of a sequence for time ``t``, and ``I`` is the size of
                the input and is equal to ``in_size``.

        Returns:
            tuple: This function returns a tuple containing three elements,
            ``hy``, ``cy`` and ``ys``.

            - ``hy`` is an updated hidden states whose shape is the same as
              ``hx``.
            - ``cy`` is an updated cell states whose shape is the same as
              ``cx``.
            - ``ys`` is a list of :class:`~chainer.Variable` . Each element
              ``ys[t]`` holds hidden states of the last layer corresponding
              to an input ``xs[t]``. Its shape is ``(L_t, N)`` for
              uni-directional LSTM and ``(L_t, 2N)`` for bi-directional LSTM
              where ``L_t`` is the length of a sequence for time ``t``,
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

    .. warning::

       ``use_cudnn`` argument is not supported anymore since v2.
       Instead, use ``chainer.using_config('use_cudnn', use_cudnn)``.
       See :func:`chainer.using_config`.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.
        initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 2.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should be 1.

    .. seealso::
        :func:`chainer.functions.n_step_lstm`

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

    .. warning::

       ``use_cudnn`` argument is not supported anymore since v2.
       Instead, use ``chainer.using_config('use_cudnn', use_cudnn)``.
       See :func:`chainer.using_config`.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.
        initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 2.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should be 1.

    .. seealso::
        :func:`chainer.functions.n_step_bilstm`

    """

    use_bi_direction = True

    def rnn(self, *args):
        return rnn.n_step_bilstm(*args)

    @property
    def n_cells(self):
        return 2
