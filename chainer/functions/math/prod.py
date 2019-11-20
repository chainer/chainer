import numpy
import six

from chainer import backend
from chainer import function_node
import chainer.functions
from chainer.utils import type_check


class Prod(function_node.FunctionNode):
    """Product of array elements over a given axis."""

    keepdims = False

    def __init__(self, axis=None, keepdims=False):
        if axis is None:
            self.axis = None
        elif isinstance(axis, six.integer_types):
            self.axis = (axis,)
        elif isinstance(axis, tuple) and all(
                isinstance(a, six.integer_types) for a in axis):
            if len(set(axis)) != len(axis):
                raise ValueError('duplicate value in axis: ({})'.format(
                    ', '.join(map(str, axis))))
            self.axis = axis
        else:
            raise TypeError('None, int or tuple of int are required')

        self.keepdims = keepdims

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

        if self.axis is not None:
            for axis in self.axis:
                if axis >= 0:
                    type_check.expect(
                        axis < in_types[0].ndim,
                    )
                else:
                    type_check.expect(
                        -axis - 1 < in_types[0].ndim,
                    )

    def forward(self, x):
        self.retain_inputs((0,))
        xp = backend.get_array_module(*x)
        return xp.asarray(x[0].prod(axis=self.axis, keepdims=self.keepdims)),

    def backward(self, indexes, gy):
        x, = self.get_retained_inputs()
        gy, = gy
        F = chainer.functions

        if self.axis is None:
            axes = tuple(six.moves.range(x.ndim))
        else:
            axes = tuple([
                axis if axis >= 0 else axis + x.ndim
                for axis in self.axis
            ])

        if not self.keepdims:
            for axis in sorted(axes):
                gy = F.expand_dims(gy, axis=axis)

        # indices of axes that are not reduced
        axes_kept = tuple(a for a in six.moves.range(x.ndim) if a not in axes)

        n_reduced_elements = 1
        for axis in axes:
            n_reduced_elements *= x.shape[axis]
        transpose_axes = axes + axes_kept

        x = x.transpose(transpose_axes)
        transposed_shape = x.shape
        kept_shape = transposed_shape[len(axes):]
        x = x.reshape((n_reduced_elements,) + kept_shape)

        def shifted_cumprod(a):
            a, _ = F.split_axis(
                F.concat([a.xp.ones((1,) + kept_shape, a.dtype), a], 0),
                (-1,), 0)
            return F.cumprod(a, 0)

        gx = shifted_cumprod(x) * F.flip(shifted_cumprod(F.flip(x, 0)), 0)
        gx = gx.reshape(transposed_shape)
        gx = gx.transpose(list(numpy.argsort(transpose_axes)))
        gx = gx * gy
        return gx,


def prod(x, axis=None, keepdims=False):
    """Product of array elements over a given axis.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Elements to calculate the product.
        axis (None, int, or tuple of int): Axis which a product is performed.
            The default (axis = None) is perform a product over all the
            dimensions of the input array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Prod(axis, keepdims).apply((x,))[0]
