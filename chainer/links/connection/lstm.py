import six

import chainer
from chainer.functions.activation import lstm
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer import initializers
from chainer import link
from chainer.links.connection import linear
from chainer import utils
from chainer import variable


class LSTMBase(link.Chain):

    def __init__(self, in_size, out_size=None, lateral_init=None,
                 upward_init=None, bias_init=None, forget_bias_init=None):
        if out_size is None:
            out_size, in_size = in_size, None

        super(LSTMBase, self).__init__()
        if bias_init is None:
            bias_init = 0
        if forget_bias_init is None:
            forget_bias_init = 1
        self.state_size = out_size
        self.lateral_init = lateral_init
        self.upward_init = upward_init
        self.bias_init = bias_init
        self.forget_bias_init = forget_bias_init

        with self.init_scope():
            self.upward = linear.Linear(in_size, 4 * out_size, initialW=0)
            self.lateral = linear.Linear(out_size, 4 * out_size, initialW=0,
                                         nobias=True)
            if in_size is not None:
                self._initialize_params()

    def _initialize_params(self):
        lateral_init = initializers._get_initializer(self.lateral_init)
        upward_init = initializers._get_initializer(self.upward_init)
        bias_init = initializers._get_initializer(self.bias_init)
        forget_bias_init = initializers._get_initializer(self.forget_bias_init)

        for i in six.moves.range(0, 4 * self.state_size, self.state_size):
            lateral_init(self.lateral.W.array[i:i + self.state_size, :])
            upward_init(self.upward.W.array[i:i + self.state_size, :])

        a, i, f, o = lstm._extract_gates(
            self.upward.b.array.reshape(1, 4 * self.state_size, 1))

        bias_init(a)
        bias_init(i)
        forget_bias_init(f)
        bias_init(o)


class StatelessLSTM(LSTMBase):

    """Stateless LSTM layer.

    This is a fully-connected LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.lstm` function, this chain holds upward and
    lateral connections as child links. This link doesn't keep cell and
    hidden states.

    Args:
        in_size (int or None): Dimension of input vectors. If ``None``,
            parameter initialization will be deferred until the first forward
            data pass at which time the size will be determined.
        out_size (int): Dimensionality of output vectors.

    Attributes:
        upward (chainer.links.Linear): Linear layer of upward connections.
        lateral (chainer.links.Linear): Linear layer of lateral connections.

    .. admonition:: Example

        There are several ways to make a StatelessLSTM link.

        Let a two-dimensional input array :math:`x`, a cell state array
        :math:`h`, and the output array of the previous step :math:`h` be:

        >>> x = np.zeros((1, 10), dtype=np.float32)
        >>> c = np.zeros((1, 20), dtype=np.float32)
        >>> h = np.zeros((1, 20), dtype=np.float32)

        1. Give both ``in_size`` and ``out_size`` arguments:

            >>> l = L.StatelessLSTM(10, 20)
            >>> c_new, h_new = l(c, h, x)
            >>> c_new.shape
            (1, 20)
            >>> h_new.shape
            (1, 20)

        2. Omit ``in_size`` argument or fill it with ``None``:

            The below two cases are the same.

            >>> l = L.StatelessLSTM(20)
            >>> c_new, h_new = l(c, h, x)
            >>> c_new.shape
            (1, 20)
            >>> h_new.shape
            (1, 20)

            >>> l = L.StatelessLSTM(None, 20)
            >>> c_new, h_new = l(c, h, x)
            >>> c_new.shape
            (1, 20)
            >>> h_new.shape
            (1, 20)

    """

    def forward(self, c, h, x):
        """Returns new cell state and updated output of LSTM.

        Args:
            c (~chainer.Variable): Cell states of LSTM units.
            h (~chainer.Variable): Output at the previous time step.
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
            ``c_new`` represents new cell state, and ``h_new`` is updated
            output of LSTM units.

        """
        if self.upward.W.array is None:
            in_size = x.size // x.shape[0]
            with chainer.using_device(self.device):
                self.upward._initialize_params(in_size)
                self._initialize_params()

        lstm_in = self.upward(x)
        if h is not None:
            lstm_in += self.lateral(h)
        if c is None:
            xp = self.xp
            with chainer.using_device(self.device):
                c = variable.Variable(
                    xp.zeros((x.shape[0], self.state_size), dtype=x.dtype))
        return lstm.lstm(c, lstm_in)


class LSTM(LSTMBase):

    """Fully-connected LSTM layer.

    This is a fully-connected LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.lstm` function, which is defined as a stateless
    activation function, this chain holds upward and lateral connections as
    child links.

    It also maintains *states*, including the cell state and the output
    at the previous time step. Therefore, it can be used as a *stateful LSTM*.

    This link supports variable length inputs. The mini-batch size of the
    current input must be equal to or smaller than that of the previous one.
    The mini-batch size of ``c`` and ``h`` is determined as that of the first
    input ``x``.
    When mini-batch size of ``i``-th input is smaller than that of the previous
    input, this link only updates ``c[0:len(x)]`` and ``h[0:len(x)]`` and
    doesn't change the rest of ``c`` and ``h``.
    So, please sort input sequences in descending order of lengths before
    applying the function.

    Args:
        in_size (int): Dimension of input vectors. If it is ``None`` or
            omitted, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        out_size (int): Dimensionality of output vectors.
        lateral_init: A callable that takes :ref:`ndarray` and edits its value.
            It is used for initialization of the lateral connections.
            May be ``None`` to use default initialization.
        upward_init: A callable that takes :ref:`ndarray` and edits its value.
            It is used for initialization of the upward connections.
            May be ``None`` to use default initialization.
        bias_init: A callable that takes :ref:`ndarray` and edits its value
            It is used for initialization of the biases of cell input,
            input gate and output gate.and gates of the upward connection.
            May be a scalar, in that case, the bias is
            initialized by this value.
            If it is ``None``, the cell-input bias is initialized to zero.
        forget_bias_init: A callable that takes :ref:`ndarray` and edits its
            value. It is used for initialization of the biases of the forget
            gate of the upward connection.
            May be a scalar, in that case, the bias is
            initialized by this value.
            If it is ``None``, the forget bias is initialized to one.


    Attributes:
        upward (~chainer.links.Linear): Linear layer of upward connections.
        lateral (~chainer.links.Linear): Linear layer of lateral connections.
        c (~chainer.Variable): Cell states of LSTM units.
        h (~chainer.Variable): Output at the previous time step.

    .. admonition:: Example

        There are several ways to make a LSTM link.

        Let a two-dimensional input array :math:`x` be:

        >>> x = np.zeros((1, 10), dtype=np.float32)

        1. Give both ``in_size`` and ``out_size`` arguments:

            >>> l = L.LSTM(10, 20)
            >>> h_new = l(x)
            >>> h_new.shape
            (1, 20)

        2. Omit ``in_size`` argument or fill it with ``None``:

            The below two cases are the same.

            >>> l = L.LSTM(20)
            >>> h_new = l(x)
            >>> h_new.shape
            (1, 20)

            >>> l = L.LSTM(None, 20)
            >>> h_new = l(x)
            >>> h_new.shape
            (1, 20)
    """

    def __init__(self, in_size, out_size=None, lateral_init=None,
                 upward_init=None, bias_init=None, forget_bias_init=None):
        if out_size is None:
            in_size, out_size = None, in_size
        super(LSTM, self).__init__(
            in_size, out_size, lateral_init, upward_init, bias_init,
            forget_bias_init)
        self.reset_state()

    def device_resident_accept(self, visitor):
        super(LSTM, self).device_resident_accept(visitor)
        if self.c is not None:
            visitor.visit_variable(self.c)
        if self.h is not None:
            visitor.visit_variable(self.h)

    def set_state(self, c, h):
        """Sets the internal state.

        It sets the :attr:`c` and :attr:`h` attributes.

        Args:
            c (~chainer.Variable): A new cell states of LSTM units.
            h (~chainer.Variable): A new output at the previous time step.

        """
        assert isinstance(c, variable.Variable)
        assert isinstance(h, variable.Variable)
        c.to_device(self.device)
        h.to_device(self.device)
        self.c = c
        self.h = h

    def reset_state(self):
        """Resets the internal state.

        It sets ``None`` to the :attr:`c` and :attr:`h` attributes.

        """
        self.c = self.h = None

    def forward(self, x):
        """Updates the internal state and returns the LSTM outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.

        """
        if self.upward.W.array is None:
            with chainer.using_device(self.device):
                in_size = utils.size_of_shape(x.shape[1:])
                self.upward._initialize_params(in_size)
                self._initialize_params()

        batch = x.shape[0]
        lstm_in = self.upward(x)
        h_rest = None
        if self.h is not None:
            h_size = self.h.shape[0]
            if batch == 0:
                h_rest = self.h
            elif h_size < batch:
                msg = ('The batch size of x must be equal to or less than'
                       'the size of the previous state h.')
                raise TypeError(msg)
            elif h_size > batch:
                h_update, h_rest = split_axis.split_axis(
                    self.h, [batch], axis=0)
                lstm_in += self.lateral(h_update)
            else:
                lstm_in += self.lateral(self.h)
        if self.c is None:
            with chainer.using_device(self.device):
                self.c = variable.Variable(
                    self.xp.zeros((batch, self.state_size), dtype=x.dtype))
        self.c, y = lstm.lstm(self.c, lstm_in)

        if h_rest is None:
            self.h = y
        elif len(y.array) == 0:
            self.h = h_rest
        else:
            self.h = concat.concat([y, h_rest], axis=0)

        return y
