import numpy
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import function
from chainer import function_node
from chainer.utils import type_check
import chainerx


def _extract_gates(x):
    r = x.reshape((len(x), x.shape[1] // 4, 4) + x.shape[2:])
    return [r[:, :, i] for i in six.moves.range(4)]


def _sigmoid(x, xp=numpy):
    half = x.dtype.type(0.5)
    return xp.tanh(x * half) * half + half


def _grad_sigmoid(x):
    return x * (1 - x)


def _grad_grad_sigmoid(x):
    return x * (1 - x) * (1 - 2 * x)


def _grad_tanh(x):
    return 1 - x * x


def _grad_grad_tanh(x, gx):
    return -2 * x * gx


_preamble = '''
template <typename T> __device__ T sigmoid(T x) {
    const T half = 0.5;
    return tanh(x * half) * half + half;
}
template <typename T> __device__ T grad_sigmoid(T y) { return y * (1 - y); }
template <typename T> __device__ T grad_tanh(T y) { return 1 - y * y; }

#define COMMON_ROUTINE \
    T aa = tanh(a); \
    T ai = sigmoid(i_); \
    T af = sigmoid(f); \
    T ao = sigmoid(o);
'''


class LSTM(function_node.FunctionNode):

    """Long short-term memory unit with forget gate.

    It has two inputs (c, x) and two outputs (c, h), where c indicates the cell
    state. x must have four times channels compared to the number of units.

    """

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('c', 'x'))
        c_type, x_type = in_types

        type_check.expect(
            c_type.dtype.kind == 'f',
            x_type.dtype == c_type.dtype,

            c_type.ndim >= 2,
            x_type.ndim >= 2,
            c_type.ndim == x_type.ndim,

            x_type.shape[0] <= c_type.shape[0],
            x_type.shape[1] == 4 * c_type.shape[1],
        )
        for i in six.moves.range(2, type_check.eval(c_type.ndim)):
            type_check.expect(x_type.shape[i] == c_type.shape[i])

    def forward_chainerx(self, inputs):
        c, x = inputs
        c_next, h = chainerx.lstm(c, x)
        return c_next, h

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        c_prev, x = inputs
        a, i, f, o = _extract_gates(x)
        batch = len(x)

        if isinstance(x, chainer.get_cpu_array_types()):
            if intel64.should_use_ideep('>=auto'):
                xp = intel64.ideep.get_array_module(x)
            else:
                xp = numpy
            a = xp.tanh(a)
            i = _sigmoid(i, xp)
            f = _sigmoid(f, xp)
            o = _sigmoid(o, xp)

            c_next = numpy.empty_like(c_prev)
            c_next[:batch] = a * i + f * c_prev[:batch]
            h = o * xp.tanh(c_next[:batch])
        else:
            c_next = cuda.cupy.empty_like(c_prev)
            h = cuda.cupy.empty_like(c_next[:batch])
            cuda.elementwise(
                'T c_prev, T a, T i_, T f, T o', 'T c, T h',
                '''
                    COMMON_ROUTINE;
                    c = aa * ai + af * c_prev;
                    h = ao * tanh(c);
                ''',
                'lstm_fwd', preamble=_preamble)(
                    c_prev[:batch], a, i, f, o, c_next[:batch], h)

        c_next[batch:] = c_prev[batch:]
        self.retain_outputs((0,))
        return c_next, h

    def backward(self, indexes, grads):
        grad_inputs = (
            self.get_retained_inputs() + self.get_retained_outputs() + grads)
        return LSTMGrad()(*grad_inputs)


class LSTMGrad(function.Function):

    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)
        c_prev, x, c_next, gc, gh = inputs
        batch = len(x)

        gx = xp.empty_like(x)
        ga, gi, gf, go = _extract_gates(gx)

        # Consider the case that either gradient is not given
        if gc is None:
            gc_update = 0
            gc_rest = 0
        else:
            gc_update = gc[:batch]
            gc_rest = gc[batch:]
        if gh is None:
            gh = 0

        a, i, f, o = _extract_gates(x)
        if xp is numpy:
            if intel64.should_use_ideep('>=auto'):
                xp = intel64.ideep.get_array_module(x)
            tanh_a = xp.tanh(a)
            sig_i = _sigmoid(i, xp)
            sig_f = _sigmoid(f, xp)
            sig_o = _sigmoid(o, xp)

            co = xp.tanh(c_next[:batch])
            gc_prev = numpy.empty_like(c_prev)
            # multiply f later
            gc_prev[:batch] = gh * sig_o * _grad_tanh(co) + gc_update
            gc = gc_prev[:batch]
            ga[:] = gc * sig_i * _grad_tanh(tanh_a)
            gi[:] = gc * tanh_a * _grad_sigmoid(sig_i)
            gf[:] = gc * c_prev[:batch] * _grad_sigmoid(sig_f)
            go[:] = gh * co * _grad_sigmoid(sig_o)
            gc_prev[:batch] *= sig_f  # multiply f here
            gc_prev[batch:] = gc_rest
        else:
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
                    c_prev[:batch], c_next[:batch], gc_update, gh, a, i, f, o,
                    gc_prev[:batch], ga, gi, gf, go)
            gc_prev[batch:] = gc_rest

        return gc_prev, gx

    def backward(self, inputs, grads):
        xp = backend.get_array_module(*inputs)

        c_prev, x, c, gc, gh = inputs
        ggc_prev, ggx = grads
        batch = len(x)

        gc_is_none = gc is None
        gh_is_none = gh is None
        ggc_prev_is_none = ggc_prev is None
        ggx_is_none = ggx is None

        if gc_is_none:
            gc = 0
        if gh_is_none:
            gh = 0
        if ggc_prev_is_none:
            ggc_prev = 0
        if ggx_is_none:
            ggx = 0

        gc_prev = xp.empty_like(c_prev)
        gx = xp.empty_like(x)
        gc_next = xp.empty_like(c)
        ggc = xp.empty_like(c_prev)
        ggh = xp.empty_like(c[:batch])

        gc_prev[batch:] = 0
        gc_next[batch:] = 0
        ggc[batch:] = 0 if ggc_prev_is_none else ggc_prev[batch:]
        ggh[batch:] = 0

        c_prev = c_prev[:batch]
        c = c[:batch]
        if not gc_is_none:
            gc = gc[:batch]
        if not ggc_prev_is_none:
            ggc_prev = ggc_prev[:batch]
        if not ggx_is_none:
            ggx = ggx[:batch]

        a, i, f, o = _extract_gates(x)
        if not ggx_is_none:
            gga, ggi, ggf, ggo = _extract_gates(ggx)
        else:
            gga = 0
            ggi = 0
            ggf = 0
            ggo = 0
        ga, gi, gf, go = _extract_gates(gx)

        lstm_grad_grad(
            c_prev, a, i, f, o, c, gc, gh, ggc_prev, gga, ggi, ggf, ggo,
            gc_prev[:batch], ga[:], gi[:], gf[:], go[:], gc_next[:batch],
            ggc[:batch], ggh[:batch])

        if gc_is_none:
            ggc = None
        if gh_is_none:
            ggh = None

        return gc_prev, gx, gc_next, ggc, ggh


@cuda.fuse()
def lstm_grad_grad(
        c_prev, a, i, f, o, c, gc, gh, ggc_prev, gga, ggi, ggf, ggo,
        gc_prev, ga, gi, gf, go, gc_next, ggc, ggh):
    xp = backend.get_array_module(a)
    sig_o = _sigmoid(o, xp)
    gsig_o = _grad_sigmoid(sig_o)
    ggsig_o = _grad_grad_sigmoid(sig_o)
    sig_i = _sigmoid(i, xp)
    gsig_i = _grad_sigmoid(sig_i)
    ggsig_i = _grad_grad_sigmoid(sig_i)
    sig_f = _sigmoid(f, xp)
    gsig_f = _grad_sigmoid(sig_f)
    ggsig_f = _grad_grad_sigmoid(sig_f)
    tanh_a = xp.tanh(a)
    gtanh_a = _grad_tanh(tanh_a)
    ggtanh_a = _grad_grad_tanh(tanh_a, gtanh_a)
    tanh_c = xp.tanh(c)
    gtanh_c = _grad_tanh(tanh_c)
    ggtanh_c = _grad_grad_tanh(tanh_c, gtanh_c)

    gc_bar = gh * sig_o * gtanh_c + gc

    gc_prev[:] = ggf * gc_bar * gsig_f
    ga[:] = (gga * sig_i * ggtanh_a + ggi * gtanh_a * gsig_i) * gc_bar
    gi[:] = (gga * gtanh_a * gsig_i + ggi * tanh_a * ggsig_i) * gc_bar
    gf[:] = (ggc_prev * (gh * sig_o * gtanh_c + gc) * gsig_f +
             ggf * gc_bar * c_prev * ggsig_f)

    ggc[:] = (ggc_prev * sig_f +
              gga * sig_i * gtanh_a +
              ggi * tanh_a * gsig_i +
              ggf * c_prev * gsig_f)

    dgc_do = gh * gsig_o * gtanh_c
    go[:] = ggc * dgc_do + ggo * gh * tanh_c * ggsig_o
    dgc_dc = gh * sig_o * ggtanh_c
    gc_next[:] = ggc * dgc_dc + ggo * gh * gtanh_c * gsig_o
    ggh[:] = ggc * sig_o * gtanh_c + ggo * tanh_c * gsig_o
    return gc_prev, ga, gi, gf, go, gc_next, ggc, ggh


def lstm(c_prev, x):
    """Long Short-Term Memory units as an activation function.

    This function implements LSTM units with forget gates. Let the previous
    cell state ``c_prev`` and the input array ``x``.

    First, the input array ``x`` is split into four arrays
    :math:`a, i, f, o` of the same shapes along the second axis. It means that
    ``x`` 's second axis must have 4 times the ``c_prev`` 's second axis.

    The split input arrays are corresponding to:

        - :math:`a` : sources of cell input
        - :math:`i` : sources of input gate
        - :math:`f` : sources of forget gate
        - :math:`o` : sources of output gate

    Second, it computes the updated cell state ``c`` and the outgoing signal
    ``h`` as:

    .. math::

        c &= \\tanh(a) \\sigma(i)
           + c_{\\text{prev}} \\sigma(f), \\\\
        h &= \\tanh(c) \\sigma(o),

    where :math:`\\sigma` is the elementwise sigmoid function.
    These are returned as a tuple of two variables.

    This function supports variable length inputs. The mini-batch size of
    the current input must be equal to or smaller than that of the previous
    one. When mini-batch size of ``x`` is smaller than that of ``c``, this
    function only updates ``c[0:len(x)]`` and doesn't change the rest of ``c``,
    ``c[len(x):]``.
    So, please sort input sequences in descending order of lengths before
    applying the function.

    Args:
        c_prev (:class:`~chainer.Variable` or :ref:`ndarray`):
            Variable that holds the previous cell state. The cell state
            should be a zero array or the output of the previous call of LSTM.
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Variable that holds the sources of cell input, input gate, forget
            gate and output gate. It must have the second dimension whose size
            is four times of that of the cell state.

    Returns:
        tuple: Two :class:`~chainer.Variable` objects ``c`` and ``h``.
        ``c`` is the updated cell state. ``h`` indicates the outgoing signal.

    See the original paper proposing LSTM with forget gates:
    `Long Short-Term Memory in Recurrent Neural Networks
    <http://www.felixgers.de/papers/phd.pdf>`_.

    .. seealso::
        :class:`~chainer.links.LSTM`

    .. admonition:: Example

        Assuming ``y`` is the current incoming signal, ``c`` is the previous
        cell state, and ``h`` is the previous outgoing signal from an ``lstm``
        function. Each of ``y``, ``c`` and ``h`` has ``n_units`` channels.
        Most typical preparation of ``x`` is:

        >>> n_units = 100
        >>> y = chainer.Variable(np.zeros((1, n_units), np.float32))
        >>> h = chainer.Variable(np.zeros((1, n_units), np.float32))
        >>> c = chainer.Variable(np.zeros((1, n_units), np.float32))
        >>> model = chainer.Chain()
        >>> with model.init_scope():
        ...   model.w = L.Linear(n_units, 4 * n_units)
        ...   model.v = L.Linear(n_units, 4 * n_units)
        >>> x = model.w(y) + model.v(h)
        >>> c, h = F.lstm(c, x)

        It corresponds to calculate the input array ``x``, or the input
        sources :math:`a, i, f, o`, from the current incoming signal ``y`` and
        the previous outgoing signal ``h``. Different parameters are used for
        different kind of input sources.

    .. note::

        We use the naming rule below.

        - incoming signal
            The formal input of the formulation of LSTM (e.g. in NLP, word
            vector or output of lower RNN layer). The input of
            :class:`chainer.links.LSTM` is the *incoming signal*.
        - input array
            The array which is linear transformed from *incoming signal* and
            the previous outgoing signal. The *input array* contains four
            sources, the sources of cell input, input gate, forget gate and
            output gate. The input of
            :class:`chainer.functions.activation.lstm.LSTM` is the
            *input array*.

    """
    return LSTM().apply((c_prev, x))
