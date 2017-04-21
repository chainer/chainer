from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import split_axis
from chainer.functions.array import concat
from chainer.links.connection import linear

from chainer import link
from chainer import utils


class ChildSumTreeLSTM(link.Chain):
    """Child-Sum TreeLSTM unit.

    This is a Child-Sum TreeLSTM unit as a chain.
    This link is a variable arguments function, which compounds
    the states of all children nodes into the new states of
    a current (parent) node. *states* denotes the cell state, c,
    and the output, h, which are produced by this link.
    This link doesn't keep cell and hidden states internally.

    Args:
        in_size (int): Dimension of input vectors.
        out_size (int): Dimensionality of cell and output vectors.

    Attributes:
        W_x (chainer.links.Linear): Linear layer of
            connections from input vectors.
        W_h_aio (chainer.links.Linear): Linear layer of connections between
            (:math:`a`, :math:`i`, :math:`o`) and summation of children's
            output vectors. :math:`a`, :math:`i` and :math:`o` denotes
            input compound (:math:`u` in the paper),
            input gate and output gate, respectively.
        W_h_f (chainer.links.Linear): Linear layer of connections between
            :math:`f` and the output of each child.

    See the paper for details: Improved Semantic Representations From
    Tree-Structured Long Short-Term Memory Networks
    <http://www.aclweb.org/anthology/P15-1150>`_.

    """

    def __init__(self, in_size, out_size):
        super(ChildSumTreeLSTM, self).__init__(
            W_x=linear.Linear(in_size, 4 * out_size),
            W_h_aio=linear.Linear(out_size, 3 * out_size, nobias=True),
            W_h_f=linear.Linear(out_size, out_size, nobias=True),
        )
        self.out_size = out_size
        utils.experimental('chainer.links.tree_lstm.py')

    def __call__(self, *cshsx):
        """Returns new cell state and output of Child-Sum TreeLSTM.

        Args:
            cshsx (list of ~chainer.Variable): Variable arguments which include
                all cell vectors and all output vectors of variable children,
                and an input vector. For example, this link is called such as
                `func(c1, c2, h1, h2, x)` if the number of children nodes is 2,
                while `func(c1, c2, c3, h1, h2, h3, x)` if that is 3.
                This function is independent from an order of children nodes.
                Thus, the returns of `func(c1, c2, h1, h2, x)` equals to
                those of `func(c2, c1, h2, h1, x)`.

        Returns:
            tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
                ``c_new`` represents new cell state vector, and ``h_new`` is
                new output vector.

        """

        cs = cshsx[:len(cshsx) // 2]
        hs = cshsx[len(cshsx) // 2:-1]
        x = cshsx[-1]
        units = self.out_size

        pre_x = self.W_x(x)
        pre_h_aio = self.W_h_aio(sum(hs))
        pre_h_fs = split_axis.split_axis(
            self.W_h_f(concat.concat(hs, axis=0)), len(hs), axis=0)

        aio = pre_x[:, :3 * units] + pre_h_aio
        fs = [pre_x[:, 3 * units:] + pre_h_f
              for pre_h_f in pre_h_fs]

        c = sigmoid.sigmoid(aio[:, units:2 * units]) * \
            tanh.tanh(aio[:, :units]) + \
            sum(sigmoid.sigmoid(f) * c for f, c in zip(fs, cs))
        h = sigmoid.sigmoid(aio[:, 2 * units:]) * tanh.tanh(c)
        return c, h


class NaryTreeLSTM(link.Chain):
    """N-ary TreeLSTM unit.

    This is a N-ary TreeLSTM unit as a chain.
    This link is a fixed-length arguments function, which compounds
    the states of all children nodes into the new states of
    a current (parent) node. *states* denotes the cell state, c,
    and the output, h, which are produced by this link.
    This link doesn't keep cell and hidden states internally.

    Args:
        in_size (int): Dimension of input vectors.
        out_size (int): Dimensionality of cell and output vectors.
        n_ary (int): The number of children nodes in a tree structure.

    Attributes:
        W_x (chainer.links.Linear): Linear layer of
            connections from input vectors.
        W_h_aio (chainer.links.Linear): Linear layer of connections between
            (:math:`a`, :math:`i`, :math:`o`, all :math:`f`s)
            and the output of each child
            :math:`a`, :math:`i`, :math:`o` and :math:`f` denotes input compund
            (:math:`u` in the paper), input gate, output gate and forget gate,
            respectively.

    See the paper for details: Improved Semantic Representations From
    Tree-Structured Long Short-Term Memory Networks
    <http://www.aclweb.org/anthology/P15-1150>`_.

    """

    def __init__(self, in_size, out_size, n_ary=2):
        assert(n_ary >= 2)
        super(NaryTreeLSTM, self).__init__(
            W_x=linear.Linear(in_size, 4 * out_size),
        )
        for i in range(1, n_ary + 1):
            self.add_link(
                'W_h{}'.format(i),
                linear.Linear(out_size, out_size * (3 + n_ary), nobias=True))
        self.out_size = out_size
        self.n_ary = n_ary
        utils.experimental('chainer.links.tree_lstm.py')

    def __call__(self, *cshsx):
        """Returns new cell state and output of N-ary TreeLSTM.

        Args:
            cshsx (list of ~chainer.Variable): Arguments which include all cell
                vectors and all output vectors of fixed-length children,
                and an input vector. The number of arguments must be same
                as ``n_ary * 2 + 1``. For example, this link is called such as
                `func(c1, c2, h1, h2, x)` if the number of children nodes
                was set 2 (``n_ary = 2``), while
                `func(c1, c2, c3, h1, h2, h3, x)` if that was 3
                (``n_ary = 3``).
                This function is dependent from an order of children nodes
                unlike Child-Sum TreeLSTM.
                Thus, the returns of `func(c1, c2, h1, h2, x)` are
                different from those of `func(c2, c1, h2, h1, x)`.

        Returns:
            tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
                ``c_new`` represents new cell state vector, and ``h_new`` is
                new output vector.

        """

        assert(len(cshsx) == self.n_ary * 2 + 1)
        cs = cshsx[:self.n_ary]
        hs = cshsx[self.n_ary:-1]
        x = cshsx[-1]
        units = self.out_size

        pre_x = self.W_x(x)
        pre_hs = [getattr(self, 'W_h{}'.format(i))(h)
                  for i, h in enumerate(hs, start=1)]
        pre_h_aio = sum(pre_h[:, :3 * units] for pre_h in pre_hs)
        pre_h_fs = split_axis.split_axis(
            sum(pre_h[:, 3 * units:] for pre_h in pre_hs), self.n_ary, axis=1)

        aio = pre_x[:, :3 * units] + pre_h_aio
        fs = [pre_x[:, 3 * units:] + pre_h_f
              for pre_h_f in pre_h_fs]

        c = sigmoid.sigmoid(aio[:, units:2 * units]) * \
            tanh.tanh(aio[:, :units]) + \
            sum(sigmoid.sigmoid(f) * c for f, c in zip(fs, cs))
        h = sigmoid.sigmoid(aio[:, 2 * units:]) * tanh.tanh(c)
        return c, h
