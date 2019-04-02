import numpy
import six

import chainer
from chainer.functions.array import permutate
from chainer.functions.array import transpose_sequence
from chainer.functions.connection import n_step_rnn as rnn
from chainer.initializers import normal
from chainer import link
from chainer.utils import argument
from chainer import variable


def argsort_list_descent(lst):
    return numpy.argsort([-len(x) for x in lst]).astype(numpy.int32)


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
    """__init__(self, n_layers, in_size, out_size, dropout)

    Base link class for Stacked RNN/BiRNN links.

    This link is base link class for :func:`chainer.links.NStepRNN` and
    :func:`chainer.links.NStepBiRNN`.

    This link's behavior depends on argument, ``use_bi_direction``.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.

    .. seealso::
        :func:`chainer.links.NStepRNNReLU`
        :func:`chainer.links.NStepRNNTanh`
        :func:`chainer.links.NStepBiRNNReLU`
        :func:`chainer.links.NStepBiRNNTanh`

    """

    def __init__(self, n_layers, in_size, out_size, dropout, **kwargs):
        if kwargs:
            argument.check_unexpected_kwargs(
                kwargs,
                use_cudnn='use_cudnn argument is not supported anymore. '
                'Use chainer.using_config',
                use_bi_direction='use_bi_direction is not supported anymore',
                activation='activation is not supported anymore')
            argument.assert_kwargs_empty(kwargs)

        weights = []
        if self.use_bi_direction:
            direction = 2
        else:
            direction = 1

        for i in six.moves.range(n_layers):
            for di in six.moves.range(direction):
                weight = link.Link()
                with weight.init_scope():
                    for j in six.moves.range(self.n_weights):
                        if i == 0 and j < self.n_weights // 2:
                            w_in = in_size
                        elif i > 0 and j < self.n_weights // 2:
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

        self.ws = [[getattr(layer, 'w%d' % i)
                    for i in six.moves.range(self.n_weights)]
                   for layer in self]
        self.bs = [[getattr(layer, 'b%d' % i)
                    for i in six.moves.range(self.n_weights)]
                   for layer in self]

        self.n_layers = n_layers
        self.dropout = dropout
        self.out_size = out_size
        self.direction = direction

    def copy(self, mode='share'):
        ret = super(NStepRNNBase, self).copy(mode)
        ret.ws = [[getattr(layer, 'w%d' % i)
                   for i in six.moves.range(ret.n_weights)] for layer in ret]
        ret.bs = [[getattr(layer, 'b%d' % i)
                   for i in six.moves.range(ret.n_weights)] for layer in ret]
        return ret

    def init_hx(self, xs):
        shape = (self.n_layers * self.direction, len(xs), self.out_size)
        with chainer.using_device(self.device):
            hx = variable.Variable(self.xp.zeros(shape, dtype=xs[0].dtype))
        return hx

    def rnn(self, *args):
        """Calls RNN function.

        This function must be implemented in a child class.
        """
        raise NotImplementedError

    @property
    def n_cells(self):
        """Returns the number of cells.

        This function must be implemented in a child class.
        """
        return NotImplementedError

    def forward(self, hx, xs, **kwargs):
        """forward(self, hx, xs)

        Calculate all hidden states and cell states.

        Args:
            hx (:class:`~chainer.Variable` or None): Initial hidden states.
                If ``None`` is specified zero-vector is used.
                Its shape is ``(S, B, N)`` for uni-directional RNN
                and ``(2S, B, N)`` for bi-directional RNN where ``S`` is
                the number of layers and is equal to ``n_layers``, ``B`` is
                the mini-batch size, and ``N`` is the dimension of
                the hidden units.
            xs (list of :class:`~chainer.Variable`): List of input sequences.
                Each element ``xs[i]`` is a :class:`chainer.Variable` holding
                a sequence. Its shape is ``(L_i, I)``, where ``L_t`` is the
                length of a sequence for batch ``i``, and ``I`` is the size of
                the input and is equal to ``in_size``.

        Returns:
            tuple: This function returns a tuple containing three elements,
            ``hy`` and ``ys``.

            - ``hy`` is an updated hidden states whose shape is same as ``hx``.
            - ``ys`` is a list of :class:`~chainer.Variable` . Each element
              ``ys[i]`` holds hidden states of the last layer corresponding
              to an input ``xs[i]``. Its shape is ``(L_i, N)`` for
              uni-directional RNN and ``(L_i, 2N)`` for bi-directional RNN
              where ``L_t`` is the length of a sequence for batch ``i``,
              and ``N`` is size of hidden units.
        """
        (hy,), ys = self._call([hx], xs, **kwargs)
        return hy, ys

    def _call(self, hs, xs, **kwargs):
        """Calls RNN function.

        Args:
            hs (list of ~chainer.Variable or None): Lisit of hidden states.
                Its length depends on its implementation.
                If ``None`` is specified zero-vector is used.
            xs (list of ~chainer.Variable): List of input sequences.
                Each element ``xs[i]`` is a :class:`chainer.Variable` holding
                a sequence.

        Returns:
            tuple: hs
        """
        if kwargs:
            argument.check_unexpected_kwargs(
                kwargs, train='train argument is not supported anymore. '
                'Use chainer.using_config')
            argument.assert_kwargs_empty(kwargs)

        assert isinstance(xs, (list, tuple))
        indices = argsort_list_descent(xs)

        xs = permutate_list(xs, indices, inv=False)
        hxs = []
        for hx in hs:
            if hx is None:
                hx = self.init_hx(xs)
            else:
                hx = permutate.permutate(hx, indices, axis=1, inv=False)
            hxs.append(hx)

        trans_x = transpose_sequence.transpose_sequence(xs)

        args = [self.n_layers, self.dropout] + hxs + \
               [self.ws, self.bs, trans_x]
        result = self.rnn(*args)

        hys = [permutate.permutate(h, indices, axis=1, inv=True)
               for h in result[:-1]]
        trans_y = result[-1]
        ys = transpose_sequence.transpose_sequence(trans_y)
        ys = permutate_list(ys, indices, inv=True)

        return hys, ys


class NStepRNNTanh(NStepRNNBase):
    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Uni-directional RNN for sequences.

    This link is stacked version of Uni-directional RNN for sequences.
    Note that the activation function is ``tanh``.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_rnn`, this function automatically
    sort inputs in descending order by length, and transpose the sequence.
    Users just need to call the link with a list of :class:`chainer.Variable`
    holding sequences.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.

    .. seealso::
        :func:`chainer.functions.n_step_rnn`

    """

    n_weights = 2
    use_bi_direction = False

    def rnn(self, *args):
        return rnn.n_step_rnn(*args, activation='tanh')

    @property
    def n_cells(self):
        return 1


class NStepRNNReLU(NStepRNNBase):
    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Uni-directional RNN for sequences.

    This link is stacked version of Uni-directional RNN for sequences.
    Note that the activation function is ``relu``.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_rnn`, this function automatically
    sort inputs in descending order by length, and transpose the sequence.
    Users just need to call the link with a list of :class:`chainer.Variable`
    holding sequences.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.

    .. seealso::
        :func:`chainer.functions.n_step_rnn`

    """

    n_weights = 2
    use_bi_direction = False

    def rnn(self, *args):
        return rnn.n_step_rnn(*args, activation='relu')

    @property
    def n_cells(self):
        return 1


class NStepBiRNNTanh(NStepRNNBase):
    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Bi-directional RNN for sequences.

    This link is stacked version of Bi-directional RNN for sequences.
    Note that the activation function is ``tanh``.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_birnn`, this function automatically
    sort inputs in descending order by length, and transpose the sequence.
    Users just need to call the link with a list of :class:`chainer.Variable`
    holding sequences.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.

    .. seealso::
        :func:`chainer.functions.n_step_birnn`

    """

    n_weights = 2
    use_bi_direction = True

    def rnn(self, *args):
        return rnn.n_step_birnn(*args, activation='tanh')

    @property
    def n_cells(self):
        return 1


class NStepBiRNNReLU(NStepRNNBase):
    """__init__(self, n_layers, in_size, out_size, dropout)

    Stacked Bi-directional RNN for sequences.

    This link is stacked version of Bi-directional RNN for sequences.
    Note that the activation function is ``relu``.
    It calculates hidden and cell states of all layer at end-of-string,
    and all hidden states of the last layer for each time.

    Unlike :func:`chainer.functions.n_step_birnn`, this function automatically
    sort inputs in descending order by length, and transpose the sequence.
    Users just need to call the link with a list of :class:`chainer.Variable`
    holding sequences.

    Args:
        n_layers (int): Number of layers.
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of hidden states and output vectors.
        dropout (float): Dropout ratio.

    .. seealso::
        :func:`chainer.functions.n_step_birnn`

    """

    n_weights = 2
    use_bi_direction = True

    def rnn(self, *args):
        return rnn.n_step_birnn(*args, activation='relu')

    @property
    def n_cells(self):
        return 1
