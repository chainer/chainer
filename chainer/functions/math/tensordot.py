import numpy
import six

from chainer import backend
from chainer import function_node
from chainer import utils
from chainer.utils import collections_abc
from chainer.utils import type_check


def _tensordot(a, b, a_axes, b_axes, c_axes=None):
    a_col_ndim = len(a_axes[1])
    b_row_ndim = len(b_axes[0])
    if a_col_ndim != b_row_ndim:
        raise ValueError('axes count mismatch')
    if a.ndim < a_col_ndim or b.ndim < b_row_ndim:
        raise ValueError('dimension of input tensors must be '
                         'greater equal to dot-axes count ({})'
                         .format(a_col_ndim))
    for a_axis, b_axis in zip(a_axes[1], b_axes[0]):
        if a.shape[a_axis] != b.shape[b_axis]:
            raise ValueError('shape mismatch')

    xp = backend.get_array_module(a)
    y = xp.tensordot(a, b, axes=(tuple(a_axes[1]), tuple(b_axes[0])))

    if c_axes is not None:
        a_row_ndim = len(a_axes[0])
        b_col_ndim = len(b_axes[1])
        c_row_ndim = len(c_axes[0])
        c_col_ndim = len(c_axes[1])
        if a_row_ndim != c_row_ndim:
            raise ValueError('axes count mismatch')
        if b_col_ndim != c_col_ndim:
            raise ValueError('axes count mismatch')

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

    def __init__(self, axes=2, a_axes=None, b_axes=None, c_axes=None,
                 dtype=None):
        self.axes = axes
        self.a_axes = a_axes
        self.b_axes = b_axes
        self.c_axes = c_axes
        self.dtype = dtype

        if isinstance(axes, collections_abc.Sequence):
            if len(axes) != 2:
                raise ValueError('axes must be a pair of sequence of integers '
                                 'when it is a list or tuple.')
        elif isinstance(axes, six.integer_types):
            pass
        else:
            raise TypeError('axes must be a pair of sequence of integers or '
                            'an integer')

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('a', 'b'))
        a_type, b_type = in_types

        type_check.expect(
            a_type.dtype.kind == 'f',
            b_type.dtype.kind == 'f',
        )

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        a, b = inputs

        if self.a_axes is None or self.b_axes is None:
            a_axes = [[], []]  # 0:row axes, 1:col axes
            b_axes = [[], []]  # 0:row axes, 1:col axes
            axes = self.axes
            if isinstance(axes, collections_abc.Sequence):
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

        c = _tensordot(a, b, self.a_axes, self.b_axes, self.c_axes)

        if self.c_axes is None:
            c_axes = [[], []]  # 0:row axes, 1:col axes
            c_row_ndim = len(self.a_axes[0])
            c_col_ndim = len(self.b_axes[1])
            c_axes[0] = six.moves.range(c_row_ndim)
            c_axes[1] = six.moves.range(c_row_ndim, c_row_ndim + c_col_ndim)
            self.c_axes = c_axes

        return utils.force_array(c, self.dtype),

    def backward(self, indexes, grad_outputs):
        a, b = self.get_retained_inputs()
        gc, = grad_outputs

        ga = None
        if 0 in indexes:
            ga, = TensorDot(a_axes=self.c_axes,
                            b_axes=[self.b_axes[1], self.b_axes[0]],
                            c_axes=self.a_axes,
                            dtype=a.dtype).apply((gc, b))

        gb = None
        if 1 in indexes:
            gb, = TensorDot(a_axes=[self.a_axes[1], self.a_axes[0]],
                            b_axes=self.c_axes,
                            c_axes=self.b_axes,
                            dtype=b.dtype).apply((a, gc))

        return ga, gb


def tensordot(a, b, axes=2):
    """Returns the tensor dot product of two arrays along specified axes.

    This is equivalent to compute dot product along the specified axes which
    are treated as one axis by reshaping.

    Args:
        a (:class:`~chainer.Variable` or :ref:`ndarray`): The first argument.
        b (:class:`~chainer.Variable` or :ref:`ndarray`): The second argument.
        axes:
            - If it is an integer, then ``axes`` axes at the last of ``a`` and
              the first of ``b`` are used.
            - If it is a pair of sequences of integers, then these two
              sequences specify the list of axes for ``a`` and ``b``. The
              corresponding axes are paired for sum-product.

    Returns:
        ~chainer.Variable: The tensor dot product of ``a`` and ``b`` along the
        axes specified by ``axes``.

    .. admonition:: Example

        >>> a = np.random.rand(5, 3, 2)
        >>> b = np.random.rand(3, 2, 4)
        >>> c = F.tensordot(a, b, axes=2)
        >>> c.shape
        (5, 4)

    .. seealso:: :func:`numpy.tensordot`

    """
    return TensorDot(axes=axes).apply((a, b))[0]
