from chainer import function
from chainer.utils import type_check


class ThinStackSet(function.Function):

    """Set values to a thin stack."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        s_type, i_type, v_type = in_types
        type_check.expect(
            s_type.dtype.kind == 'f',
            i_type.dtype.kind == 'i',
            s_type.dtype == v_type.dtype,
            s_type.ndim == 3,
            i_type.ndim == 1,
            v_type.ndim == 2,
            s_type.shape[0] >= i_type.shape[0],
            i_type.shape[0] == v_type.shape[0],
            s_type.shape[2] == v_type.shape[1],
        )

    def forward(self, inputs):
        stack, indices, values = inputs
        stack[range(len(indices)), indices] = values
        return stack,

    def backward(self, inputs, grads):
        _, indices, _ = inputs
        g = grads[0]
        gv = g[range(len(indices)), indices]
        g[range(len(indices)), indices] = 0
        return g, None, gv


def thin_stack_set(s, i, x):
    return ThinStackSet()(s, i, x)


class ThinStackGet(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        s_type, i_type = in_types
        type_check.expect(
            s_type.dtype.kind == 'f',
            i_type.dtype.kind == 'i',
            s_type.ndim == 3,
            i_type.ndim == 1,
            s_type.shape[0] >= i_type.shape[0],
        )

    def forward(self, inputs):
        stack, indices = inputs
        return stack[range(len(indices)), indices], stack

    def backward(self, inputs, grads):
        stack, indices = inputs
        g, gs = grads
        if gs is None:
            gs = numpy.zeros_like(stack)
        if g is not None:
            gs[range(len(indices)), indices] += g
        return gs, None


def thin_stack_get(s, i):
    return ThinStackGet()(s, i)
