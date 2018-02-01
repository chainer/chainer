import collections
import numpy
import six

from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


def _tensordot(a, b, a_axes, b_axes, c_axes):

    a_row_ndim = len(a_axes[0])
    a_col_ndim = len(a_axes[1])
    b_row_ndim = len(b_axes[0])
    b_col_ndim = len(b_axes[1])
    c_row_ndim = len(c_axes[0])
    c_col_ndim = len(c_axes[1])
    if a_col_ndim != b_row_ndim:
        raise ValueError('axes count mismatch')
    if a_row_ndim != c_row_ndim:
        raise ValueError('axes count mismatch')
    if b_col_ndim != c_col_ndim:
        raise ValueError('axes count mismatch')
    if a.ndim < a_col_ndim or b.ndim < b_row_ndim:
        raise ValueError('dimension of input tensors must be '
                         'greater equal to dot-axes count ({})'
                         .format(a_col_ndim))
    for a_axis, b_axis in zip(a_axes[1], b_axes[0]):
        if a.shape[a_axis] != b.shape[b_axis]:
            raise ValueError('shape mismatch')

    xp = cuda.get_array_module(a)
    y = xp.tensordot(a, b, axes=(a_axes[1], b_axes[0]))

    trans = [None for i in six.moves.range(y.ndim)]
    table_a = [1 if i in a_axes[0] else 0 for i in six.moves.range(a.ndim)]
    table_a = numpy.cumsum(table_a) - 1
    for i, c_axis in enumerate(c_axes[0]):
        trans[c_axis] = table_a[a_axes[0][i]]
    table_b = [1 if i in b_axes[1] else 0 for i in six.moves.range(b.ndim)]
    table_b = numpy.cumsum(table_b) - 1
    for i, c_axis in enumerate(c_axes[1]):
        trans[c_axis] = table_b[b_axes[1][i]] + len(a_axes[0])
    for i, c_axis in enumerate(trans):
        if i != c_axis:
            y = xp.transpose(y, trans)
            break

    return y


class TensorDot(function_node.FunctionNode):

    def __init__(self, axes=2, a_axes=None, b_axes=None, c_axes=None):
        self.axes = axes
        self.a_axes = a_axes
        self.b_axes = b_axes
        self.c_axes = c_axes

        if isinstance(axes, collections.Sequence):
            if len(axes) != 2:
                raise ValueError('axes must consist of two arrays.')
        elif isinstance(axes, int):
            pass
        else:
            raise TypeError('axes must be int or tuple of int')

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        a_type, b_type = in_types

        type_check.expect(
            a_type.dtype.kind == 'f',
            b_type.dtype.kind == 'f',
            a_type.ndim >= 1,
            b_type.ndim >= 1,
        )

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        a, b = inputs

        if self.a_axes is None or self.b_axes is None:
            a_axes = [[], []]  # 0:row axes, 1:col axes
            b_axes = [[], []]  # 0:row axes, 1:col axes
            axes = self.axes
            if isinstance(axes, collections.Sequence):
                a_axes[1], b_axes[0] = axes
                if numpy.isscalar(a_axes[1]):
                    a_axes[1] = a_axes[1],
                if numpy.isscalar(b_axes[0]):
                    b_axes[0] = b_axes[0],
            else:
                a_axes[1] = six.moves.range(a.ndim - axes, a.ndim)
                b_axes[0] = six.moves.range(axes)
            a_range = six.moves.range(a.ndim)
            a_axes[0] = [i for i in a_range if i not in a_axes[1]]
            b_range = six.moves.range(b.ndim)
            b_axes[1] = [i for i in b_range if i not in b_axes[0]]
            self.a_axes = a_axes
            self.b_axes = b_axes

        if self.c_axes is None:
            c_axes = [[], []]  # 0:row axes, 1:col axes
            c_row_ndim = len(self.a_axes[0])
            c_col_ndim = len(self.b_axes[1])
            c_axes[0] = six.moves.range(c_row_ndim)
            c_axes[1] = six.moves.range(c_row_ndim, c_row_ndim + c_col_ndim)
            self.c_axes = c_axes

        c = _tensordot(a, b, self.a_axes, self.b_axes, self.c_axes)
        return utils.force_array(c),

    def backward(self, indexes, grad_outputs):
        a, b = self.get_retained_inputs()
        gc, = grad_outputs

        ga = None
        if 0 in indexes:
            ga, = TensorDot(a_axes=self.c_axes,
                            b_axes=[self.b_axes[1], self.b_axes[0]],
                            c_axes=self.a_axes).apply((gc, b))
            if ga.dtype != a.dtype:
                ga.array = ga.array.astype(a.dtype)

        gb = None
        if 1 in indexes:
            gb, = TensorDot(a_axes=[self.a_axes[1], self.a_axes[0]],
                            b_axes=self.c_axes,
                            c_axes=self.b_axes).apply((a, gc))
            if gb.dtype != b.dtype:
                gb.array = gb.array.astype(b.dtype)

        return ga, gb


def tensordot(a, b, axes=2):

    return TensorDot(axes=axes).apply((a, b))[0]
