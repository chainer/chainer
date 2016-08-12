from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer import link
from chainer.links.connection import linear
from chainer.utils import rnn
from chainer import variable


class PeepholeLSTM(link.Chain):

    state_names = ('c', 'h')

    def __init__(self, in_size, out_size):
        super(PeepholeLSTM, self).__init__(
            upward=linear.Linear(in_size, 4 * out_size),
            lateral=linear.Linear(out_size, 4 * out_size, nobias=True),
            peep_i=linear.Linear(out_size, out_size, nobias=True),
            peep_f=linear.Linear(out_size, out_size, nobias=True),
            peep_o=linear.Linear(out_size, out_size, nobias=True),
        )
        self.state_shapes = ((out_size,), (out_size,))

    def __call__(self, c, h, x):
        """Updates the internal state and returns the LSTM outputs.

        Args:
            c (~chainer.Variable): Cell states of LSTM units.
            h (~chainer.Variable): Output at the current time step.
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.

        """
        lstm_in = self.upward(x)
        if h is not None:
            lstm_in += self.lateral(h)
        if c is None:
            xp = self.xp
            c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        lstm_in = reshape.reshape(lstm_in, (len(lstm_in.data),
                                            lstm_in.data.shape[1] // 4,
                                            4))
        a, i, f, o = split_axis.split_axis(lstm_in, 4, 2)
        a = reshape.reshape(a, (len(a.data), a.data.shape[1]))
        i = reshape.reshape(i, (len(i.data), i.data.shape[1]))
        f = reshape.reshape(f, (len(f.data), f.data.shape[1]))
        o = reshape.reshape(o, (len(o.data), o.data.shape[1]))
        peep_in_i = self.peep_i(c)
        peep_in_f = self.peep_f(c)
        a = tanh.tanh(a)
        i = sigmoid.sigmoid(i + peep_in_i)
        f = sigmoid.sigmoid(f + peep_in_f)
        c = a * i + f * c
        peep_in_o = self.peep_o(c)
        o = sigmoid.sigmoid(o + peep_in_o)
        h = o * tanh.tanh(c)
        return c, h


StatefulPeepholeLSTMBase = rnn.create_stateful_rnn(
    PeepholeLSTM, 'StatefulPeepholeLSTMBase')


class StatefulPeepholeLSTM(StatefulPeepholeLSTMBase):

    """Fully-connected LSTM layer with peephole connections.

    This is a fully-connected LSTM layer with peephole connections as a chain.
    Unlike the :link:`~chainer.links.lstm` link, this chain holds ``peep_i``,
    ``peep_f`` and ``peep_o`` as child links besides ``upward`` and
    ``lateral``.

    Given a input vector :math:`x`, Peephole returns the next hidden vector
    :math:`h'` defined as

    .. math::

       a &=& \\tanh(upward x + lateral h), \\\\
       i &=& \\sigma(upward x + lateral h + peep_i c), \\\\
       f &=& \\sigma(upward x + lateral h + peep_f c), \\\\
       c' &=& a \\odot i + f \\odot c, \\\\
       o &=& \\sigma(upward x + lateral h + peep_o c'), \\\\
       h' &=& o \\tanh(c'),

    where :math:`\\sigma` is the sigmoid function, :math:`\\odot` is the
    element-wise product, :math:`c` is the current cell state, :math:`c'`
    is the next cell state and :math:`h` is the current hidden vector.

    Args:
        in_size(int): Dimension of the input vector :math:`x`.
        out_size(int): Dimension of the hidden vector :math: `h`.

    Attributes:
        upward (~chainer.links.Linear): Linear layer of upward connections.
        lateral (~chainer.links.Linear): Linear layer of lateral connections.
        peep_i (~chainer.links.Linear): Linear layer of peephole connections
                                        to the input gate.
        peep_f (~chainer.links.Linear): Linear layer of peephole connections
                                        to the forget gate.
        peep_o (~chainer.links.Linear): Linear layer of peephole connections
                                        to the output gate.

    """

    pass
