import numpy
import six

from chainer import cuda
from chainer.functions.array import permutate
from chainer.functions.array import transpose_sequence
from chainer.functions.connection import n_step_gru as rnn
from chainer.initializers import normal
from chainer import link
from chainer.links.connection.n_step_rnn import argsort_list_descent
from chainer.links.connection.n_step_rnn import permutate_list
from chainer.utils import argument
from chainer import variable


class NStepGRUBase(link.ChainList):

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

    def __init__(self, n_layers, in_size, out_size, dropout, use_bi_direction,
                 **kwargs):
        argument.check_unexpected_kwargs(
            kwargs, use_cudnn='use_cudnn argument is not supported anymore. '
            'Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)

        weights = []
        direction = 2 if use_bi_direction else 1
        for i in six.moves.range(n_layers):
            for di in six.moves.range(direction):
                weight = link.Link()
                with weight.init_scope():
                    for j in six.moves.range(6):
                        if i == 0 and j < 3:
                            w_in = in_size
                        elif i > 0 and j < 3:
                            w_in = out_size * direction
                        else:
                            w_in = out_size
                        w = variable.Parameter(
                            normal.Normal(numpy.sqrt(1. / w_in)),
                            (out_size, w_in))
                        b = variable.Parameter(0, (out_size,))
                        setattr(weight, 'w%d' % j, w)
                        setattr(weight, 'b%d' % j, b)
                weights.append(weight)

        super(NStepGRUBase, self).__init__(*weights)

        self.n_layers = n_layers
        self.dropout = dropout
        self.out_size = out_size
        self.direction = direction
        self.rnn = rnn.n_step_bigru if use_bi_direction else rnn.n_step_gru

    def init_hx(self, xs):
        shape = (self.n_layers * self.direction, len(xs), self.out_size)
        with cuda.get_device_from_id(self._device_id):
            hx = variable.Variable(self.xp.zeros(shape, dtype=xs[0].dtype))
        return hx

    def __call__(self, hx, xs, **kwargs):
        """__call__(self, hx, xs)

        Calculate all hidden states and cell states.

        .. warning::

           ``train`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', train)``.
           See :func:`chainer.using_config`.

        Args:
            hx (~chainer.Variable or None): Initial hidden states. If ``None``
                is specified zero-vector is used.
            xs (list of ~chianer.Variable): List of input sequences.
                Each element ``xs[i]`` is a :class:`chainer.Variable` holding
                a sequence.

        """
        argument.check_unexpected_kwargs(
            kwargs, train='train argument is not supported anymore. '
            'Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)

        assert isinstance(xs, (list, tuple))
        indices = argsort_list_descent(xs)

        xs = permutate_list(xs, indices, inv=False)
        if hx is None:
            hx = self.init_hx(xs)
        else:
            hx = permutate.permutate(hx, indices, axis=1, inv=False)

        trans_x = transpose_sequence.transpose_sequence(xs)

        ws = [[w.w0, w.w1, w.w2, w.w3, w.w4, w.w5] for w in self]
        bs = [[w.b0, w.b1, w.b2, w.b3, w.b4, w.b5] for w in self]

        hy, trans_y = self.rnn(
            self.n_layers, self.dropout, hx, ws, bs, trans_x)

        hy = permutate.permutate(hy, indices, axis=1, inv=True)
        ys = transpose_sequence.transpose_sequence(trans_y)
        ys = permutate_list(ys, indices, inv=True)

        return hy, ys


class NStepGRU(NStepGRUBase):

    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Uni-directional GRU for sequnces.

    This link is stacked version of Uni-directional GRU for sequences.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_gru`, this function automatically
    sort inputs in descending order by length, and transpose the seuqnece.
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

    def __init__(self, n_layers, in_size, out_size, dropout, **kwargs):
        NStepGRUBase.__init__(
            self, n_layers, in_size, out_size, dropout,
            use_bi_direction=False, **kwargs)


class NStepBiGRU(NStepGRUBase):

    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Bi-directional GRU for sequnces.

    This link is stacked version of Bi-directional GRU for sequences.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_bigru`, this function automatically
    sort inputs in descending order by length, and transpose the seuqnece.
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

    def __init__(self, n_layers, in_size, out_size, dropout, **kwargs):
        NStepGRUBase.__init__(
            self, n_layers, in_size, out_size, dropout,
            use_bi_direction=True, **kwargs)
