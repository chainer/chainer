import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _extract_gates(x):
    """Extract gates by split.

    This is a different from ``_extract_gates`` in lstm.py,
    which is as follows
    ```
        r = x.reshape((x.shape[0], x.shape[1] // 4, 4) + x.shape[2:])
        return (r[:, :, i] for i in six.moves.range(4))
    ```
    In other words, it thinly slices x and merge them,
    while this thickly slices x.

    """
    r = x.reshape((x.shape[0], 5, x.shape[1] // 5) + x.shape[2:])
    return (r[:, i, :] for i in six.moves.range(5))


def _sigmoid(x):
    half = x.dtype.type(0.5)
    return numpy.tanh(x * half) * half + half


def _grad_sigmoid(x):
    return x * (1 - x)


def _grad_tanh(x):
    return 1 - x * x


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
    T ao = sigmoid(o); \
    T af1 = sigmoid(f1); \
    T af2 = sigmoid(f2);
'''


class NaryTreeLSTM(function.Function):

    """N-ary TreeLSTM unit with two forget gates.

    Modified from Tai et al. (arxiv:1503.00075) and exactly as in Bowman et al.
    (arxiv:1603.06021); we have three inputs (c1, c2, x) where x is 5 times
    larger in the feature dimension and represents everything inside the
    activation functions. This means 15/14 as many independent parameters
    as Tai; in particular, f1 and f2 can depend in different ways on
    the LSTM input. There are two outputs (c, h).

    """

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        c1_type, c2_type, x_type = in_types

        type_check.expect(
            c1_type.dtype.kind == 'f',
            c2_type.dtype.kind == 'f',
            x_type.dtype == c1_type.dtype,
            x_type.dtype == c2_type.dtype,

            c1_type.ndim >= 2,
            c2_type.ndim >= 2,
            x_type.ndim >= 2,
            c1_type.ndim == x_type.ndim,
            c2_type.ndim == x_type.ndim,

            x_type.shape[0] == c1_type.shape[0],
            x_type.shape[0] == c2_type.shape[0],
            x_type.shape[1] == 5 * c1_type.shape[1],
            x_type.shape[1] == 5 * c2_type.shape[1],
        )

        for i in six.moves.range(2, type_check.eval(c1_type.ndim)):
            type_check.expect(x_type.shape[i] == c1_type.shape[i])
        for i in six.moves.range(2, type_check.eval(c2_type.ndim)):
            type_check.expect(x_type.shape[i] == c2_type.shape[i])

    def forward(self, inputs):
        c_prev1, c_prev2, x = inputs
        a, i, o, f1, f2 = _extract_gates(x)

        if isinstance(x, numpy.ndarray):
            self.a = numpy.tanh(a)
            self.i = _sigmoid(i)
            self.o = _sigmoid(o)
            self.f1 = _sigmoid(f1)
            self.f2 = _sigmoid(f2)

            self.c = self.a * self.i + self.f1 * c_prev1 + self.f2 * c_prev2
            h = self.o * numpy.tanh(self.c)
        else:
            self.c, h = cuda.elementwise(
                'T c_prev1, T c_prev2, T a, T i_, T o, T f1, T f2', 'T c, T h',
                '''
                    COMMON_ROUTINE;
                    c = aa * ai + af1 * c_prev1 + af2 * c_prev2;
                    h = ao * tanh(c);
                ''',
                'treelstm_fwd', preamble=_preamble)(
                    c_prev1, c_prev2, a, i, o, f1, f2)

        return self.c, h

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        c_prev1, c_prev2, x = inputs
        gc, gh = grad_outputs

        gx = xp.empty_like(x)
        ga, gi, go, gf1, gf2 = _extract_gates(gx)

        # Consider the case that either gradient is not given
        if gc is None:
            gc = 0
        if gh is None:
            gh = 0

        if xp is numpy:
            co = numpy.tanh(self.c)
            tmp = gh * self.o * _grad_tanh(co) + gc
            ga[:] = tmp * self.i * _grad_tanh(self.a)
            gi[:] = tmp * self.a * _grad_sigmoid(self.i)
            go[:] = gh * co * _grad_sigmoid(self.o)
            gf1[:] = tmp * c_prev1 * _grad_sigmoid(self.f1)
            gf2[:] = tmp * c_prev2 * _grad_sigmoid(self.f2)
            gc_prev1 = tmp * self.f1
            gc_prev2 = tmp * self.f2
        else:
            a, i, o, f1, f2 = _extract_gates(x)
            gc_prev1 = xp.empty_like(c_prev1)
            gc_prev2 = xp.empty_like(c_prev2)
            cuda.elementwise(
                'T c_prev1, T c_prev2, T c, T gc, T gh, T a, T i_, T o, '
                'T f1, T f2',
                'T gc_prev1, T gc_prev2, T ga, T gi, T go, T gf1, T gf2',
                '''
                    COMMON_ROUTINE;
                    T co = tanh(c);
                    T temp = gh * ao * grad_tanh(co) + gc;
                    ga = temp * ai * grad_tanh(aa);
                    gi = temp * aa * grad_sigmoid(ai);
                    go = gh * co * grad_sigmoid(ao);
                    gf1 = temp * c_prev1 * grad_sigmoid(af1);
                    gf2 = temp * c_prev2 * grad_sigmoid(af2);
                    gc_prev1 = temp * af1;
                    gc_prev2 = temp * af2;
                ''',
                'treelstm_bwd', preamble=_preamble)(
                    c_prev1, c_prev2, self.c, gc, gh, a, i, o, f1, f2,
                    gc_prev1, gc_prev2, ga, gi, go, gf1, gf2)

        return gc_prev1, gc_prev2, gx


def n_ary_tree_lstm(c_prev1, c_prev2, x):
    """N-ary Tree-LSTM unit as an activation function.

    This function implements N-ary Tree-LSTM units, which is proposed
    by Tai et al. and modified by Bowman et al. Let the
    previous cell states :math:`c_{\\text{prev1}}` :math:`c_{\\text{prev2}}`
    and the incoming signal :math:`x`.

    First, the incoming signal :math:`x` is split into five arrays
    :math:`a, i, o, f1, f2` of the same shapes along the second axis.
    It means that :math:`x` 's second axis must have 5 times the length of
    each :math:`c_{\\text{prev}}`.

    The splitted input signals are corresponding to:

        - :math:`a` : sources of cell input
        - :math:`i` : sources of input gate
        - :math:`o` : sources of output gate
        - :math:`f1` : sources of forget gate 1
        - :math:`f2` : sources of forget gate 2

    Second, it computes outputs as:

    .. math::

        c &= \\tanh(a) \\text{sigmoid}(i)
           + c_{\\text{prev1}} \\text{sigmoid}(f1),
           + c_{\\text{prev2}} \\text{sigmoid}(f2), \\\\
        h &= \\tanh(c) \\text{sigmoid}(o).

    These are returned as a tuple of two variables.

    Args:
        c_prev1 (~chainer.Variable): Variable that holds the first child cell
            state. The cell state should be a zero array or the output of the
            previous call of LSTM.
        c_prev2 (~chainer.Variable): Variable that holds the second child cell
            state. The cell state should be a zero array or the output of the
            previous call of LSTM.
        x (~chainer.Variable): Variable that holds the incoming signal. It must
            have the second dimension five times of that of each cell state,

    Returns:
        tuple: Two :class:`~chainer.Variable` objects ``c`` and ``h``. ``c`` is
            the updated cell state. ``h`` indicates the outgoing signal.

    See Tai et al. paper's proposal for N-Ary Tree-LSTM (Sec. 3.2, but note
        that Eq. 10 only has one W matrix, applied to x, for all children,
        while we have one for each, as shown in Bowman et al. paper):
    `Improved Semantic Representations From Tree-Structured Long \
    Short-Term Memory Networks \
    <http://arxiv.org/pdf/1503.00075v3.pdf>`_.
    `A Fast Unified Model for Parsing and Sentence Understanding \
    <https://arxiv.org/pdf/1603.06021.pdf>`_.

    .. admonition:: Example

        Assuming ``y`` is the current input signal, ``c`` is the previous cell
        state, and ``h`` is the previous output signal from an
        ``n_ary_tree_lstm``
        function. Each of ``y``, ``c`` and ``h`` has ``n_units`` channels.
        Most typical preparation of ``x`` is:

        >>> model = FunctionSet(w=F.Linear(n_units, 5 * n_units),
        ...                     v1=F.Linear(n_units, 5 * n_units),
        ...                     v2=F.Linear(n_units, 5 * n_units),
        ...                     ...)
        >>> x = model.w(y) + model.v1(h1) + model.v2(h2)
        >>> c, h = F.n_ary_tree_lstm(c1, c2, x)

        It corresponds to calculate the input sources :math:`a, i, o, f1, f2`
        from the current input ``y`` and the children's outputs
        ``h1`` and ``h2``. Different parameters are used for different kind of
        input sources.

    """
    return NaryTreeLSTM()(c_prev1, c_prev2, x)
