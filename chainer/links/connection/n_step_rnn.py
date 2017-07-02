import numpy
import six

from chainer import cuda
from chainer.functions.array import permutate
from chainer.functions.array import transpose_sequence
from chainer.functions.connection import n_step_rnn as rnn
from chainer.initializers import normal
from chainer import link
from chainer.utils import argument
from chainer import variable


def argsort_list_descent(lst):
    return numpy.argsort([-len(x.data) for x in lst]).astype('i')


def permutate_list(lst, indices, inv):
    ret = [None] * len(lst)
    if inv:
        for i, ind in enumerate(indices):
            ret[ind] = lst[i]
    else:
        for i, ind in enumerate(indices):
            ret[i] = lst[ind]
    return ret


class NStepRNNBase(link.ChainList):
    """__init__(self, n_layers, in_size, out_size, dropout, use_bi_direction, activation)

    Base link class for Stacked RNN/BiRNN links.

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
        use_bi_direction (bool): if ``True``, use Bi-directional RNN.
            if ``False``, use Uni-directional RNN.
        activation (str): Activation function name.
            Please select ``tanh`` or ``relu``.

    .. seealso::
        :func:`chainer.links.NStepRNNReLU`
        :func:`chainer.links.NStepRNNTanh`
        :func:`chainer.links.NStepBiRNNReLU`
        :func:`chainer.links.NStepBiRNNTanh`

    """  # NOQA

    def __init__(self, n_layers, in_size, out_size, dropout,
                 use_bi_direction, activation, **kwargs):
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
                    for j in six.moves.range(2):
                        if i == 0 and j < 1:
                            w_in = in_size
                        elif i > 0 and j < 1:
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

        super(NStepRNNBase, self).__init__(*weights)

        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation
        self.out_size = out_size
        self.direction = direction
        self.rnn = rnn.n_step_birnn if use_bi_direction else rnn.n_step_rnn

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

        ws = [[w.w0, w.w1] for w in self]
        bs = [[w.b0, w.b1] for w in self]

        hy, trans_y = self.rnn(
            self.n_layers, self.dropout, hx, ws, bs, trans_x,
            activation=self.activation)

        hy = permutate.permutate(hy, indices, axis=1, inv=True)
        ys = transpose_sequence.transpose_sequence(trans_y)
        ys = permutate_list(ys, indices, inv=True)

        return hy, ys


class NStepRNNTanh(NStepRNNBase):
    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Uni-directional RNN for sequnces.

    This link is stacked version of Uni-directional RNN for sequences.
    Note that the activation function is ``tanh``.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_rnn`, this function automatically
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
        :func:`chainer.functions.n_step_rnn`

    """

    def __init__(self, n_layers, in_size, out_size, dropout, **kwargs):
        NStepRNNBase.__init__(
            self, n_layers, in_size, out_size, dropout,
            use_bi_direction=False,  activation='tanh', **kwargs)


class NStepRNNReLU(NStepRNNBase):
    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Uni-directional RNN for sequnces.

    This link is stacked version of Uni-directional RNN for sequences.
    Note that the activation function is ``relu``.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_rnn`, this function automatically
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
        :func:`chainer.functions.n_step_rnn`

    """

    def __init__(self, n_layers, in_size, out_size, dropout, **kwargs):
        NStepRNNBase.__init__(
            self, n_layers, in_size, out_size, dropout,
            use_bi_direction=False, activation='relu', **kwargs)


class NStepBiRNNTanh(NStepRNNBase):
    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Bi-directional RNN for sequnces.

    This link is stacked version of Bi-directional RNN for sequences.
    Note that the activation function is ``tanh``.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_birnn`, this function automatically
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
        use_cudnn (bool): Use cuDNN.

    .. seealso::
        :func:`chainer.functions.n_step_birnn`

    """

    def __init__(self, n_layers, in_size, out_size, dropout, **kwargs):
        NStepRNNBase.__init__(
            self, n_layers, in_size, out_size, dropout,
            use_bi_direction=True, activation='tanh', **kwargs)


class NStepBiRNNReLU(NStepRNNBase):
    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Bi-directional RNN for sequnces.

    This link is stacked version of Bi-directional RNN for sequences.
    Note that the activation function is ``relu``.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_birnn`, this function automatically
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
        :func:`chainer.functions.n_step_birnn`

    """

    def __init__(self, n_layers, in_size, out_size, dropout, **kwargs):
        NStepRNNBase.__init__(
            self, n_layers, in_size, out_size, dropout,
            use_bi_direction=True, activation='relu', **kwargs)
