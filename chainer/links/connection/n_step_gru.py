from chainer.functions.connection import n_step_gru as rnn
from chainer.links.connection import n_step_rnn


class NStepGRUBase(n_step_rnn.NStepRNNBase):

    """__init__(self, n_layers, in_size, out_size, dropout, use_bi_direction)

    Base link class for Stacked GRU/BiGRU links.

    This link is base link class for :func:`chainer.links.NStepRNN` and
    :func:`chainer.links.NStepBiRNN`.
    This link's behavior depends on argument, ``use_bi_direction``.

    .. warning::

       ``use_cudnn`` argument is not supported anymore since v2.
       Instead, use ``chainer.using_config('use_cudnn', use_cudnn)``.
       See :func:`chainer.using_config`.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.
        use_bi_direction (bool): if ``True``, use Bi-directional GRU.
            if ``False``, use Uni-directional GRU.
    .. seealso::
        :func:`chainer.links.NStepGRU`
        :func:`chainer.links.NStepBiGRU`

    """

    n_weights = 6


class NStepGRU(NStepGRUBase):

    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Uni-directional GRU for sequences.

    This link is stacked version of Uni-directional GRU for sequences.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_gru`, this function automatically
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

    .. seealso::
        :func:`chainer.functions.n_step_gru`

    """

    use_bi_direction = False

    def rnn(self, *args):
        return rnn.n_step_gru(*args)


class NStepBiGRU(NStepGRUBase):

    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Bi-directional GRU for sequences.

    This link is stacked version of Bi-directional GRU for sequences.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_bigru`, this function automatically
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

    .. seealso::
        :func:`chainer.functions.n_step_bigru`

    """

    use_bi_direction = True

    def rnn(self, *args):
        return rnn.n_step_bigru(*args)
