import numpy
import six

from chainer import cuda
from chainer.functions.activation import lstm
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer import initializers
from chainer import link
from chainer.links.connection import linear
from chainer import variable


def _init_weight(weights, initializer):
    initializers._get_initializer(initializer)(weights)


class LSTMBase(link.Chain):

    def __init__(self, in_size, out_size=None, lateral_init=None,
                 upward_init=None, bias_init=0, forget_bias_init=1):
        if out_size is None:
            out_size, in_size = in_size, None

        super(LSTMBase, self).__init__()
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

        for i in six.moves.range(0, 4 * self.state_size, self.state_size):
            lateral_init(self.lateral.W.data[i:i + self.state_size, :])
            upward_init(self.upward.W.data[i:i + self.state_size, :])

        a, i, f, o = lstm._extract_gates(
            self.upward.b.data.reshape(1, 4 * self.state_size, 1))
        _init_weight(a, self.bias_init)
        _init_weight(i, self.bias_init)
        _init_weight(f, self.forget_bias_init)
        _init_weight(o, self.bias_init)


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

        >>> x = np.zeros((1, 10), dtype='f')
        >>> c = np.zeros((1, 20), dtype='f')
        >>> h = np.zeros((1, 20), dtype='f')

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

    def __call__(self, c, h, x):
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
        if self.upward.W.data is None:
            in_size = x.size // x.shape[0]
            with cuda.get_device_from_id(self._device_id):
                self.upward._initialize_params(in_size)
                self._initialize_params()

        lstm_in = self.upward(x)
        if h is not None:
            lstm_in += self.lateral(h)
        if c is None:
            xp = self.xp
            with cuda.get_device_from_id(self._device_id):
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
        lateral_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the lateral connections.
            May be ``None`` to use default initialization.
        upward_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the upward connections.
            May be ``None`` to use default initialization.
        bias_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value
            It is used for initialization of the biases of cell input,
            input gate and output gate.and gates of the upward connection.
            May be a scalar, in that case, the bias is
            initialized by this value.
            If it is ``None``, the cell-input bias is initialized to zero.
        forget_bias_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value
            It is used for initialization of the biases of the forget gate of
            the upward connection.
            May be a scalar, in that case, the bias is
            initialized by this value.
            If it is ``None``, the forget bias is initialized to one.


    Attributes:
        upward (~chainer.links.Linear): Linear layer of upward connections.
        lateral (~chainer.links.Linear): Linear layer of lateral connections.
        c (~chainer.Variable): Cell states of LSTM units.
        h (~chainer.Variable): Output at the previous time step.

    """

    def __init__(self, in_size, out_size=None, **kwargs):
        if out_size is None:
            in_size, out_size = None, in_size
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

    def set_state(self, c, h):
        """Sets the internal state.

        It sets the :attr:`c` and :attr:`h` attributes.

        Args:
            c (~chainer.Variable): A new cell states of LSTM units.
            h (~chainer.Variable): A new output at the previous time step.

        """
        assert isinstance(c, variable.Variable)
        assert isinstance(h, variable.Variable)
        c_ = c
        h_ = h
        if self.xp == numpy:
            c_.to_cpu()
            h_.to_cpu()
        else:
            c_.to_gpu(self._device_id)
            h_.to_gpu(self._device_id)
        self.c = c_
        self.h = h_

    def reset_state(self):
        """Resets the internal state.

        It sets ``None`` to the :attr:`c` and :attr:`h` attributes.

        """
        self.c = self.h = None

    def __call__(self, x):
        """Updates the internal state and returns the LSTM outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.

        """
        if self.upward.W.data is None:
            with cuda.get_device_from_id(self._device_id):
                in_size = x.size // x.shape[0]
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
            xp = self.xp
            with cuda.get_device_from_id(self._device_id):
                self.c = variable.Variable(
                    xp.zeros((batch, self.state_size), dtype=x.dtype))
        self.c, y = lstm.lstm(self.c, lstm_in)

        if h_rest is None:
            self.h = y
        elif len(y.data) == 0:
            self.h = h_rest
        else:
            self.h = concat.concat([y, h_rest], axis=0)

        return y
