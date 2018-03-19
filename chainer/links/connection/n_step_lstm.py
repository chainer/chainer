import six

from chainer.backends import cuda
from chainer.functions.array import permutate
from chainer.functions.array import transpose_sequence
from chainer.functions.connection import n_step_lstm as rnn
from chainer import initializers
from chainer import link
from chainer.links.connection import n_step_rnn
from chainer.utils import argument
from chainer import variable


class NStepLSTMBase(link.ChainList):
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

    def __init__(self, n_layers, in_size, out_size, dropout,
                 initialW, initial_bias, use_bi_direction,
                 **kwargs):
        argument.check_unexpected_kwargs(
            kwargs, use_cudnn='use_cudnn argument is not supported anymore. '
            'Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)

        if initial_bias is None:
            initial_bias = initializers.constant.Zero()
        initialW = initializers._get_initializer(initialW)

        weights = []
        direction = 2 if use_bi_direction else 1
        for i in six.moves.range(n_layers):
            for di in six.moves.range(direction):
                weight = link.Link()
                with weight.init_scope():
                    for j in six.moves.range(8):
                        if i == 0 and j < 4:
                            w_in = in_size
                        elif i > 0 and j < 4:
                            w_in = out_size * direction
                        else:
                            w_in = out_size
                        name_w = 'w{}'.format(j)
                        name_b = 'b{}'.format(j)
                        w = variable.Parameter(initialW, (out_size, w_in))
                        b = variable.Parameter(initial_bias, (out_size,))
                        setattr(weight, name_w, w)
                        setattr(weight, name_b, b)
                weights.append(weight)

        super(NStepLSTMBase, self).__init__(*weights)

        self.n_layers = n_layers
        self.dropout = dropout
        self.out_size = out_size
        self.direction = direction
        self.rnn = rnn.n_step_bilstm if use_bi_direction else rnn.n_step_lstm

    def init_hx(self, xs):
        shape = (self.n_layers * self.direction, len(xs), self.out_size)
        with cuda.get_device_from_id(self._device_id):
            hx = variable.Variable(self.xp.zeros(shape, dtype=xs[0].dtype))
        return hx

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
        argument.check_unexpected_kwargs(
            kwargs, train='train argument is not supported anymore. '
            'Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)

        assert isinstance(xs, (list, tuple))
        xp = cuda.get_array_module(hx, *xs)
        indices = n_step_rnn.argsort_list_descent(xs)
        indices_array = xp.array(indices)

        xs = n_step_rnn.permutate_list(xs, indices, inv=False)
        if hx is None:
            hx = self.init_hx(xs)
        else:
            hx = permutate.permutate(hx, indices_array, axis=1, inv=False)

        if cx is None:
            cx = self.init_hx(xs)
        else:
            cx = permutate.permutate(cx, indices_array, axis=1, inv=False)

        trans_x = transpose_sequence.transpose_sequence(xs)

        ws = [[w.w0, w.w1, w.w2, w.w3, w.w4, w.w5, w.w6, w.w7] for w in self]
        bs = [[w.b0, w.b1, w.b2, w.b3, w.b4, w.b5, w.b6, w.b7] for w in self]

        hy, cy, trans_y = self.rnn(
            self.n_layers, self.dropout, hx, cx, ws, bs, trans_x)

        hy = permutate.permutate(hy, indices_array, axis=1, inv=True)
        cy = permutate.permutate(cy, indices_array, axis=1, inv=True)
        ys = transpose_sequence.transpose_sequence(trans_y)
        ys = n_step_rnn.permutate_list(ys, indices, inv=True)

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

    def __init__(self, n_layers, in_size, out_size, dropout,
                 initialW=None, initial_bias=None, **kwargs):
        NStepLSTMBase.__init__(
            self, n_layers, in_size, out_size, dropout,
            initialW, initial_bias,
            use_bi_direction=False, **kwargs)


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

    def __init__(self, n_layers, in_size, out_size, dropout,
                 initialW=None, initial_bias=None, **kwargs):
        NStepLSTMBase.__init__(
            self, n_layers, in_size, out_size, dropout,
            initialW, initial_bias,
            use_bi_direction=True, **kwargs)
