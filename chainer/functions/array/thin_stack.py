import numpy

from chainer import cuda
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
        xp = cuda.get_array_module(*inputs)
        stack, indices, values = inputs
        if xp is numpy:
            stack[range(len(indices)), indices] = values
        else:
            cuda.elementwise(
                'S t, T v, int32 d',
                'raw T s',
                '''
                int b = i / d;
                int k = i - b * d;
                int ind[] = {b, t, k};
                s[ind] = v;
                ''',
                'thin_stack_set_fwd'
            )(indices[:, None], values, values.shape[1], stack)

        return stack,

    def backward(self, inputs, grads):
        xp = cuda.get_array_module(*inputs)
        _, indices, _ = inputs
        g = grads[0]
        if xp is numpy:
            gv = g[range(len(indices)), indices]
            g[range(len(indices)), indices] = 0
        else:
            dim = g.shape[2]
            shape = (indices.shape[0], dim)
            gv = cuda.cupy.empty(shape, g.dtype)
            cuda.elementwise(
                'S t, int32 d',
                'raw T s, T y',
                '''
                int b = i / d;
                int k = i - b * d;
                int ind[] = {b, t, k};
                y = s[ind];
                s[ind] = 0;
                ''',
                'thin_stack_set_bwd'
            )(indices[:, None], dim, g, gv)
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
        xp = cuda.get_array_module(*inputs)
        stack, indices = inputs
        if xp is numpy:
            return stack[range(len(indices)), indices], stack
        else:
            dim = stack.shape[2]
            shape = (indices.shape[0], dim)
            y = cuda.cupy.empty(shape, stack.dtype)
            cuda.elementwise(
                'S t, int32 d, raw T s',
                'T y',
                '''
                int b = i / d;
                int k = i - b * d;
                int ind[] = {b, t, k};
                y = s[ind];
                ''',
                'thin_stack_get_fwd'
            )(indices[:, None], dim, stack, y)
            return y, stack

    def backward(self, inputs, grads):
        xp = cuda.get_array_module(*inputs)
        stack, indices = inputs
        g, gs = grads
        if gs is None:
            gs = xp.zeros_like(stack)
        if g is not None:
            if xp is numpy:
                gs[range(len(indices)), indices] += g
            else:
                dim = stack.shape[2]
                cuda.elementwise(
                    'S t, int32 d',
                    'raw T gs, T g',
                    '''
                    int b = i / d;
                    int k = i - b * d;
                    int ind[] = {b, t, k};
                    gs[ind] += g;
                    ''',
                    'thin_stack_get_bwd'
                )(indices[:, None], dim, gs, g)
        return gs, None


def thin_stack_get(s, i):
    return ThinStackGet()(s, i)
