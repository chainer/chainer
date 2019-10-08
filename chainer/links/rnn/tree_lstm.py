import numpy

from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer.functions.rnn import tree_lstm
from chainer import link
from chainer.links.connection import linear


class ChildSumTreeLSTM(link.Chain):
    """Child-Sum TreeLSTM unit.

    .. warning::

        This feature is experimental. The interface can change in the future.

    This is a Child-Sum TreeLSTM unit as a chain.
    This link is a variable arguments function, which compounds
    the states of all children nodes into the new states of
    a current (parent) node. *states* denotes the cell state, :math:`c`,
    and the output, :math:`h`, which are produced by this link.
    This link doesn't keep cell and hidden states internally.

    For example, this link is called such as
    ``func(c1, c2, h1, h2, x)`` if the number of children nodes is 2,
    while ``func(c1, c2, c3, h1, h2, h3, x)`` if that is 3.
    This function is *independent* from an order of children nodes.
    Thus, the returns of ``func(c1, c2, h1, h2, x)`` equal to
    those of ``func(c2, c1, h2, h1, x)``.

    Args:
        in_size (int): Dimension of input vectors.
        out_size (int): Dimensionality of cell and output vectors.

    Attributes:
        W_x (chainer.links.Linear): Linear layer of
            connections from input vectors.
        W_h_aio (chainer.links.Linear): Linear layer of connections between
            (:math:`a`, :math:`i`, :math:`o`) and summation of children's
            output vectors. :math:`a`, :math:`i` and :math:`o` denotes
            input compound,
            input gate and output gate, respectively.
            :math:`a`, input compound, equals to :math:`u` in
            the paper by Tai et al.
        W_h_f (chainer.links.Linear): Linear layer of connections between
            forget gate :math:`f` and the output of each child.

    See the paper for details: `Improved Semantic Representations From
    Tree-Structured Long Short-Term Memory Networks
    <https://www.aclweb.org/anthology/P15-1150>`_.

    """

    def __init__(self, in_size, out_size):
        super(ChildSumTreeLSTM, self).__init__()
        with self.init_scope():
            self.W_x = linear.Linear(in_size, 4 * out_size)
            self.W_h_aio = linear.Linear(out_size, 3 * out_size, nobias=True)
            self.W_h_f = linear.Linear(out_size, out_size, nobias=True)

        self.in_size = in_size
        self.state_size = out_size

    def forward(self, *cshsx):
        """Returns new cell state and output of Child-Sum TreeLSTM.

        Args:
            cshsx (list of :class:`~chainer.Variable`): Variable arguments
                which include all cell vectors and all output vectors of
                variable children, and an input vector.

        Returns:
            tuple of ~chainer.Variable: Returns
            :math:`(c_{new}, h_{new})`, where :math:`c_{new}` represents
            new cell state vector, and :math:`h_{new}` is new output
            vector.

        """

        cs = cshsx[:len(cshsx) // 2]
        hs = cshsx[len(cshsx) // 2:-1]
        x = cshsx[-1]
        assert(len(cshsx) % 2 == 1)
        assert(len(cs) == len(hs))

        if x is None:
            if any(c is not None for c in cs):
                base = [c for c in cs if c is not None][0]
            elif any(h is not None for h in hs):
                base = [h for h in hs if h is not None][0]
            else:
                raise ValueError('All inputs (cs, hs, x) are None.')
            batchsize, dtype = base.shape[0], base.dtype
            x = self.xp.zeros(
                (batchsize, self.in_size), dtype=dtype)

        W_x_in = self.W_x(x)
        W_x_aio_in, W_x_f_in = split_axis.split_axis(
            W_x_in, [3 * self.state_size], axis=1)

        if len(hs) == 0:
            aio_in = W_x_aio_in
            a, i, o = split_axis.split_axis(aio_in, 3, axis=1)
            c = sigmoid.sigmoid(i) * tanh.tanh(a)
            h = sigmoid.sigmoid(o) * tanh.tanh(c)
            return c, h

        hs = self._pad_zero_nodes(
            hs, (x.shape[0], self.state_size), dtype=x.dtype)
        cs = self._pad_zero_nodes(
            cs, (x.shape[0], self.state_size), dtype=x.dtype)

        aio_in = self.W_h_aio(sum(hs)) + W_x_aio_in
        W_h_fs_in = concat.concat(split_axis.split_axis(
            self.W_h_f(concat.concat(hs, axis=0)), len(hs), axis=0),
            axis=1)
        f_in = W_h_fs_in + \
            concat.concat([W_x_f_in] * len(hs), axis=1)
        tree_lstm_in = concat.concat([aio_in, f_in], axis=1)

        return tree_lstm.tree_lstm(*(cs + (tree_lstm_in, )))

    def _pad_zero_nodes(self, vs, shape, dtype=numpy.float32):
        if any(v is None for v in vs):
            zero = self.xp.zeros(shape, dtype=dtype)
            return tuple(zero if v is None else v for v in vs)
        else:
            return vs


class NaryTreeLSTM(link.Chain):
    """N-ary TreeLSTM unit.

    .. warning::

        This feature is experimental. The interface can change in the future.

    This is a N-ary TreeLSTM unit as a chain.
    This link is a fixed-length arguments function, which compounds
    the states of all children nodes into the new states of
    a current (parent) node. *states* denotes the cell state, :math:`c`,
    and the output, :math:`h`, which are produced by this link.
    This link doesn't keep cell and hidden states internally.

    For example, this link is called such as
    ``func(c1, c2, h1, h2, x)`` if the number of children nodes
    was set 2 (``n_ary = 2``), while
    ``func(c1, c2, c3, h1, h2, h3, x)`` if that was 3
    (``n_ary = 3``).
    This function is *dependent* from an order of children nodes
    unlike Child-Sum TreeLSTM.
    Thus, the returns of ``func(c1, c2, h1, h2, x)`` are
    different from those of ``func(c2, c1, h2, h1, x)``.

    Args:
        in_size (int): Dimension of input vectors.
        out_size (int): Dimensionality of cell and output vectors.
        n_ary (int): The number of children nodes in a tree structure.

    Attributes:
        W_x (chainer.links.Linear): Linear layer of
            connections from input vectors.
        W_h (chainer.links.Linear): Linear layer of connections between
            (:math:`a`, :math:`i`, :math:`o`, all :math:`f`)
            and the output of each child.
            :math:`a`, :math:`i`, :math:`o` and :math:`f` denotes input
            compound, input gate, output gate and forget gate, respectively.
            :math:`a`, input compound, equals to :math:`u` in
            the paper by Tai et al.

    See the papers for details: `Improved Semantic Representations From
    Tree-Structured Long Short-Term Memory Networks
    <https://www.aclweb.org/anthology/P15-1150>`_, and
    `A Fast Unified Model for Parsing and Sentence Understanding
    <https://arxiv.org/pdf/1603.06021.pdf>`_.

    Tai et al.'s N-Ary TreeLSTM is little extended in
    Bowman et al., and this link is based on
    the variant by Bowman et al.
    Specifically, eq. 10 in Tai et al. has only one :math:`W` matrix
    to be applied to :math:`x`, consistently for all children.
    On the other hand, Bowman et al.'s model has multiple matrices,
    each of which affects the forget gate for each child's cell individually.

    """

    def __init__(self, in_size, out_size, n_ary=2):
        assert(n_ary >= 1)
        super(NaryTreeLSTM, self).__init__()
        with self.init_scope():
            self.W_x = linear.Linear(in_size, (3 + n_ary) * out_size)

            for i in range(1, n_ary + 1):
                l = linear.Linear(
                    out_size, (3 + n_ary) * out_size, nobias=True)
                setattr(self, 'W_h{}'.format(i), l)
        self.in_size = in_size
        self.state_size = out_size
        self.n_ary = n_ary

    def forward(self, *cshsx):
        """Returns new cell state and output of N-ary TreeLSTM.

        Args:
            cshsx (list of :class:`~chainer.Variable`): Arguments which include
                all cell vectors and all output vectors of fixed-length
                children, and an input vector. The number of arguments must be
                same as ``n_ary * 2 + 1``.

        Returns:
            tuple of ~chainer.Variable: Returns :math:`(c_{new}, h_{new})`,
            where :math:`c_{new}` represents new cell state vector,
            and :math:`h_{new}` is new output vector.

        """

        assert(len(cshsx) == self.n_ary * 2 + 1)
        cs = cshsx[:self.n_ary]
        hs = cshsx[self.n_ary:-1]
        x = cshsx[-1]

        if x is None:
            if any(c is not None for c in cs):
                base = [c for c in cs if c is not None][0]
            elif any(h is not None for h in hs):
                base = [h for h in hs if h is not None][0]
            else:
                raise ValueError('All inputs (cs, hs, x) are None.')
            batchsize, dtype = base.shape[0], base.dtype
            x = self.xp.zeros(
                (batchsize, self.in_size), dtype=dtype)

        tree_lstm_in = self.W_x(x)

        for i, h in enumerate(hs, start=1):
            if h is not None:
                tree_lstm_in += getattr(self, 'W_h{}'.format(i))(h)

        cs = self._pad_zero_nodes(
            cs, (x.shape[0], self.state_size), dtype=x.dtype)

        return tree_lstm.tree_lstm(*(cs + (tree_lstm_in, )))

    def _pad_zero_nodes(self, vs, shape, dtype=numpy.float32):
        if any(v is None for v in vs):
            zero = self.xp.zeros(shape, dtype=dtype)
            return tuple(zero if v is None else v for v in vs)
        else:
            return vs
