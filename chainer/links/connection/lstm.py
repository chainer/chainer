from chainer import cuda
from chainer.functions.activation import lstm
from chainer import link
from chainer.links.connection import linear
from chainer import variable


class LSTM(link.DictLink):

    """Fully-connected LSTM layer.

    This link implements a fully-connected LSTM layer. Unlike the
    :func:`~chainer.functions.lstm`, which is defined as a stateless activation
    function, this link also contains the lateral connections as *child* links.

    This link includes following child links:

    - ``input``: Linear layer from an input sequence.
    - ``lateral``: Linear layer from the previous hidden state.

    This link is an implementation of *a stateful LSTM layer*. It holds the
    previous hidden output and cell state, and uses them to process the next
    input. The internal state should be reset by calling the
    :meth:`reset_state` method before feeding the first variable of the input
    sequence.

    Args:
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of output vectors.

    """
    def __init__(self, in_size, out_size, nobias=True):
        super(LSTM, self).__init__(
            input=linear.Linear(in_size, 4 * out_size, nobias=nobias),
            lateral=linear.Linear(in_size, 4 * out_size, nobias=True),
        )
        self.state_size = out_size
        self.reset_state()

    def reset_state(self):
        """Resets the internal state of the LSTM layer."""
        self.c = None
        self.h = None

    def __call__(self, x):
        """Updates the internal state and returns the LSTM output.

        Args:
            x (~chainer.Variable): A new batch of vectors in the input
                sequence.

        Returns:
            ~chainer.Variable: Updated hidden units.

        """
        xp = cuda.get_array_module(x.data)
        if self.c is None:
            self.c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile=self.volatile)
        if self.h is None:
            self.h = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile=self.volatile)
            
        lstm_in = self['input'](x) + self['lateral'](self.h)
        self.c, self.h = lstm.lstm(self.c, lstm_in)
        return self.h
