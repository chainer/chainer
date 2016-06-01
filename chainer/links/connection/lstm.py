import six

from chainer.functions.activation import lstm
from chainer.functions.array import concat
from chainer import initializers
from chainer import link
from chainer.links.connection import linear
from chainer import variable


class LSTMBase(link.Chain):

    def __init__(self, in_size, out_size,
                 lateral_init=None, upward_init=None,
                 bias_init=0, forget_bias_init=0):
        super(LSTMBase, self).__init__(
            upward=linear.Linear(in_size, 4 * out_size, initialW=0),
            lateral=linear.Linear(out_size, 4 * out_size,
                                  initialW=0, nobias=True),
        )
        self.state_size = out_size

        for i in six.moves.range(0, 4 * out_size, out_size):
            initializers.init_weight(
                self.lateral.W.data[i:i + out_size, :], lateral_init)
            initializers.init_weight(
                self.upward.W.data[i:i + out_size, :], upward_init)

        a, i, f, o = lstm._extract_gates(
            self.upward.b.data.reshape(1, 4 * out_size, 1))
        initializers.init_weight(a, bias_init)
        initializers.init_weight(i, bias_init)
        initializers.init_weight(f, forget_bias_init)
        initializers.init_weight(o, bias_init)


class StatelessLSTM(LSTMBase):

    """Stateless LSTM layer.

    This is a fully-connected LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.lstm` function, this chain holds upward and
    lateral connections as child links. This link doesn't keep cell and
    hidden states.

    Args:
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of output vectors.

    Attributes:
        upward (chainer.links.Linear): Linear layer of upward connections.
        lateral (chainer.links.Linear): Linear layer of lateral connections.

    """

    def __call__(self, c, h, x):
        """Returns new cell state and updated output of LSTM.

        Args:
            c (~chainer.Variable): Cell states of LSTM units.
            h (~chainer.Variable): Output at the previous timestep.
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
                ``c_new`` represents new cell state, and ``h_new`` is updated
                output of LSTM units.

        """
        lstm_in = self.upward(x)
        if h is not None:
            lstm_in += self.lateral(h)
        if c is None:
            xp = self.xp
            c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        return lstm.lstm(c, lstm_in)


class LSTM(LSTMBase):

    """Fully-connected LSTM layer.

    This is a fully-connected LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.lstm` function, which is defined as a stateless
    activation function, this chain holds upward and lateral connections as
    child links.

    It also maintains *states*, including the cell state and the output
    at the previous time step. Therefore, it can be used as a *stateful LSTM*.

    Args:
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of output vectors.
        lateral_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the lateral connections.
            Maybe be ``None`` to use default initialization.
        upward_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the upward connections.
            Maybe be ``None`` to use default initialization.
        bias_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value
            It is used for initialization of the biases of cell input,
            input gate and output gate.and gates of the upward connection.
            Maybe a scalar, in that case, the bias is
            initialized by this value.
            Maybe be ``None`` to use default initialization.
        forget_bias_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value
            It is used for initialization of the biases of the forget gate of
            the upward connection.
            Maybe a scalar, in that case, the bias is
            initialized by this value.
            Maybe be ``None`` to use default initialization.


    Attributes:
        upward (~chainer.links.Linear): Linear layer of upward connections.
        lateral (~chainer.links.Linear): Linear layer of lateral connections.
        c (~chainer.Variable): Cell states of LSTM units.
        h (~chainer.Variable): Output at the previous time step.

    """

    def __init__(self, in_size, out_size, **kwargs):
        super(LSTM, self).__init__(in_size, out_size, **kwargs)
        self.reset_state()

    def to_cpu(self):
        super(LSTM, self).to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(LSTM, self).to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def reset_state(self):
        """Resets the internal state.

        It sets ``None`` to the :attr:`c` and :attr:`h` attributes.

        """
        self.c = self.h = None

    def set_state(self, c, h):
        assert isinstance(h, chainer.Variable)
        assert isinstance(c, chainer.Variable)
        h_ = h
        c_ = c
        if self.xp == numpy:
            h_.to_cpu()
            c_.to_cpu()
        else:
            h_.to_gpu()
            c_.to_gpu()
        self.h = h_
        self.c = c_

    def __call__(self, x):
        """Updates the internal state and returns the LSTM outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.

        """
        lstm_in = self.upward(x)
        if self.h is not None:
            lstm_in += self.lateral(self.h)
        if self.c is None:
            xp = self.xp
            self.c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        self.c, self.h = lstm.lstm(self.c, lstm_in)
        return self.h

class StackedStatelessLSTM(link.ChainList):

    """Stacked Stateless Long Short term Memory (LSTM).

    This is an implementation of a Stacked Stateless LSTM.
    The underlying idea is to simply stack multiple LSTMs
    where the LSTM at the bottom takes the regular input,
    and the LSTMs after that simply take the outputs
    (represented by h) of the lower LSTMs as inputs.
    Since this is a stateless implementation,
    the states of all the LSTMs must be returned
    Args:
          in_size (int)- The size of embeddings of the inputs
          out_size (int)- The size of the hidden layer representation of
                      each GRU unit
          num_layers (int)- The number of LSTM layers

    Attributes:
          num_layers: Indicates the number of LSTM layers
    User Defined Methods:

    """

    def __init__(self, in_size, out_size, num_layers=1):
        super(StackedStatelessLSTM, self).__init__()
        assert num_layers >= 1
        self.add_link(StatelessLSTM(out_size, in_size))
        for i in range(1, num_layers):
            self.add_link(StatelessLSTM(out_size, out_size))
        self.num_layers = num_layers

    def __call__(self, h, x):
        """Updates the internal state and returns the  LSTM outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.
            h (~chainer.Variable): The list of the previous cell outputs.

        Returns:
            ~chainer.Variable: A list of the outputs (h) of the updated
                LSTM units over all the layers.

        """
        h_list = []
        h = split_axis.split_axis(h, self.num_layers, 1, True)
        h_curr = x
        for layer, h in six.moves.zip(self, h):
            h_curr = layer(h, h_curr)
            h_list.append(h_curr)
        return concat.concat(h_list, 1)

class StackedStatefulLSTM(link.ChainList):

    """Fully-connected Stacked LSTM layer.

    This is a fully-connected Stacked LSTM layer as a chain.
    It simply stacks multiple LSTMs.

    It also maintains *states*, including the cell state and the output
    at the previous time step for each LSTM in the stack.
    Therefore, it can be used as a *stateful Stacked LSTM*.

    Args:
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of output vectors.
        num_layers (int): Number of LSTM layers.

    Attributes:


    """
    def __init__(self, in_size, out_size, num_layers=1):
        super(StackedStatefulLSTM, self).__init__()
        self.add_link(LSTM(in_size, out_size))
        for i in range(1, num_layers):
            self.add_link(LSTM(out_size, out_size))
        self.num_layers = num_layers
        self.reset_state()

    def to_cpu(self):
        for layer in self:
            layer.to_cpu()

    def to_gpu(self, device=None):
        for layer in self:
            layer.to_gpu(device)

    def reset_state(self):
        """Resets the internal state.

        It sets ``None`` to the :attr:`c` and :attr:`h`
        attributes for each LSTM in the stack.

        """
        for layer in self:
            layer.reset_state()

    def set_state(self, c, h):
        h = split_axis.split_axis(h, self.num_layers, 1, True)
        c = split_axis.split_axis(c, self.num_layers, 1, True)
        for layer, c, h in six.moves.zip(self, c, h):
            assert isinstance(h, chainer.Variable)
            assert isinstance(c, chainer.Variable)
            layer.set_state(c, h)

    def __call__(self, x, top_n=None):
        """Updates the internal state and returns the LSTM outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.
            top_n (int): The number of LSTMs from the top whose outputs
            you want (default: outputs of all LSTMs are returned)

        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.

        """
        if top_n is None:
            top_n = self.num_layers

        h_list = []
        h_curr = x
        for layer in self:
            h_curr = layer(h_curr)
            h_list.append(h_curr)
        return concat.concat(h_list[-top_n:], 1)
