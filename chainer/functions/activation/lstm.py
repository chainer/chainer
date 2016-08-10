import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _extract_gates(x):
    r = x.reshape((len(x), x.shape[1] // 4, 4) + x.shape[2:])
    return [r[:, :, i] for i in six.moves.range(4)]


def _sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def _grad_sigmoid(x):
    return x * (1 - x)


def _grad_tanh(x):
    return 1 - x * x


_preamble = '''
template <typename T> __device__ T sigmoid(T x) { return 1 / (1 + exp(-x)); }
template <typename T> __device__ T grad_sigmoid(T y) { return y * (1 - y); }
template <typename T> __device__ T grad_tanh(T y) { return 1 - y * y; }

#define COMMON_ROUTINE \
    T aa = tanh(a); \
    T ai = sigmoid(i_); \
    T af = sigmoid(f); \
    T ao = sigmoid(o);
'''


class LSTM(function.Function):

    """Long short-term memory unit with forget gate.

    It has two inputs (c, x) and two outputs (c, h), where c indicates the cell
    state. x must have four times channels compared to the number of units.

    """

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        c_type, x_type = in_types

        type_check.expect(
            c_type.dtype.kind == 'f',
            x_type.dtype == c_type.dtype,

            c_type.ndim >= 2,
            x_type.ndim >= 2,
            c_type.ndim == x_type.ndim,

            x_type.shape[0] == c_type.shape[0],
            x_type.shape[1] == 4 * c_type.shape[1],
        )
        for i in six.moves.range(2, c_type.ndim.eval()):
            type_check.expect(x_type.shape[i] == c_type.shape[i])

    def forward(self, inputs):
        c_prev, x = inputs
        a, i, f, o = _extract_gates(x)

        if isinstance(x, numpy.ndarray):
            self.a = numpy.tanh(a)
            self.i = _sigmoid(i)
            self.f = _sigmoid(f)
            self.o = _sigmoid(o)

            self.c = self.a * self.i + self.f * c_prev
            h = self.o * numpy.tanh(self.c)
        else:
            self.c, h = cuda.elementwise(
                'T c_prev, T a, T i_, T f, T o', 'T c, T h',
                '''
                    COMMON_ROUTINE;
                    c = aa * ai + af * c_prev;
                    h = ao * tanh(c);
                ''',
                'lstm_fwd', preamble=_preamble)(c_prev, a, i, f, o)

        return self.c, h

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        c_prev, x = inputs
        gc, gh = grad_outputs

        gx = xp.empty_like(x)
        ga, gi, gf, go = _extract_gates(gx)

        # Consider the case that either gradient is not given
        if gc is None:
            gc = 0
        if gh is None:
            gh = 0

        if xp is numpy:
            co = numpy.tanh(self.c)
            gc_prev = gh * self.o * _grad_tanh(co) + gc  # multiply f later
            ga[:] = gc_prev * self.i * _grad_tanh(self.a)
            gi[:] = gc_prev * self.a * _grad_sigmoid(self.i)
            gf[:] = gc_prev * c_prev * _grad_sigmoid(self.f)
            go[:] = gh * co * _grad_sigmoid(self.o)
            gc_prev *= self.f  # multiply f here
        else:
            a, i, f, o = _extract_gates(x)
            gc_prev = xp.empty_like(c_prev)
            cuda.elementwise(
                'T c_prev, T c, T gc, T gh, T a, T i_, T f, T o',
                'T gc_prev, T ga, T gi, T gf, T go',
                '''
                    COMMON_ROUTINE;
                    T co = tanh(c);
                    T temp = gh * ao * grad_tanh(co) + gc;
                    ga = temp * ai * grad_tanh(aa);
                    gi = temp * aa * grad_sigmoid(ai);
                    gf = temp * c_prev * grad_sigmoid(af);
                    go = gh * co * grad_sigmoid(ao);
                    gc_prev = temp * af;
                ''',
                'lstm_bwd', preamble=_preamble)(
                    c_prev, self.c, gc, gh, a, i, f, o,
                    gc_prev, ga, gi, gf, go)

        return gc_prev, gx


def lstm(c_prev, x):
    """Long Short-Term Memory units as an activation function.

    This function implements LSTM units with forget gates. Let the previous
    cell state :math:`c_{\\text{prev}}` and the incoming signal :math:`x`.

    First, the incoming signal :math:`x` is split into four arrays
    :math:`a, i, f, o` of the same shapes along the second axis.
    It means that :math:`x` 's second axis must have 4 times the length of
    :math:`c_{\\text{prev}}`.

    The split input signals are corresponding to:

        - :math:`a` : sources of cell input
        - :math:`i` : sources of input gate
        - :math:`f` : sources of forget gate
        - :math:`o` : sources of output gate

    Second, it computes outputs as:

    .. math::

        c &= \\tanh(a) \\text{sigmoid}(i)
           + c_{\\text{prev}} \\text{sigmoid}(f), \\\\
        h &= \\tanh(c) \\text{sigmoid}(o).

    These are returned as a tuple of two variables.

    Args:
        c_prev (~chainer.Variable): Variable that holds the previous cell
            state. The cell state should be a zero array or the output of the
            previous call of LSTM.
        x (~chainer.Variable): Variable that holds the incoming signal. It must
            have the second dimension four times of that of the cell state,

    Returns:
        tuple: Two :class:`~chainer.Variable` objects ``c`` and ``h``. ``c`` is
            the updated cell state. ``h`` indicates the outgoing signal.

    See the original paper proposing LSTM with forget gates:
    `Long Short-Term Memory in Recurrent Neural Networks \
    <http://www.felixgers.de/papers/phd.pdf>`_.

    .. admonition:: Example

        Assuming ``y`` is the current input signal, ``c`` is the previous cell
        state, and ``h`` is the previous output signal from an ``lstm``
        function. Each of ``y``, ``c`` and ``h`` has ``n_units`` channels.
        Most typical preparation of ``x`` is:

        >>> import chainer, chainer.functions as F
        >>> n_units = 100
        >>> y = chainer.Variable(numpy.zeros((1, n_units), 'f'))
        >>> h = chainer.Variable(numpy.zeros((1, n_units), 'f'))
        >>> c = chainer.Variable(numpy.zeros((1, n_units), 'f'))
        >>> model = chainer.Chain(w=F.Linear(n_units, 4 * n_units),
        ...                       v=F.Linear(n_units, 4 * n_units),)
        >>> x = model.w(y) + model.v(h)
        >>> c, h = F.lstm(c, x)

        It corresponds to calculate the input sources :math:`a, i, f, o` from
        the current input ``y`` and the previous output ``h``. Different
        parameters are used for different kind of input sources.

    """
    return LSTM()(c_prev, x)
