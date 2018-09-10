import numpy
import six

from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer.utils import type_check


class Prod(function_node.FunctionNode):
    """Product of array elements over a given axis."""

    keepdims = False

    def __init__(self, axis=None, keepdims=False):
        if axis is None:
            self.axis = None
        elif isinstance(axis, int):
            self.axis = (axis,)
        elif isinstance(axis, tuple) and all(isinstance(a, int) for a in axis):
            if len(set(axis)) != len(axis):
                raise ValueError('duplicate value in axis: ({})'.format(
                    ', '.join(map(str, axis))))
            self.axis = axis
        else:
            raise TypeError('None, int or tuple of int are required')

        self.keepdims = keepdims

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

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
        xp = cuda.get_array_module(*x)
        return xp.asarray(x[0].prod(axis=self.axis, keepdims=self.keepdims)),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()
        xp = cuda.get_array_module(x[0].data)

        x = x[0]
        gy = gy[0]

        if self.axis is None:
            axes = list(six.moves.range(x.ndim))
        else:
            axes = []
            for axis in self.axis:
                if axis < 0:
                    axis += len(x.shape)
                axes.append(axis)

        if not self.keepdims:
            for axis in sorted(axes):
                gy = chainer.functions.expand_dims(gy, axis=axis)

        axes = tuple(axes)
        # indices of axes that are not reduced
        axes_kept = tuple(a for a in six.moves.range(x.ndim) if a not in axes)

        n_reduced_elements = 1
        for axis in axes:
            n_reduced_elements *= x.shape[axis]
        n_output_elements = x.size // n_reduced_elements
        transpose_axes = axes_kept + axes

        x = x.transpose(transpose_axes)
        transposed_shape = x.shape
        x = x.reshape(-1, n_reduced_elements)
        extended_x = chainer.functions.repeat(x, n_reduced_elements, 0)
        mask = xp.tile(xp.arange(n_reduced_elements), n_output_elements)
        cond = xp.zeros_like(extended_x.data, dtype=xp.bool_)
        cond[xp.arange(x.size), mask] = 1
        extended_x = chainer.functions.where(
            cond, cond.astype(extended_x.dtype), extended_x)

        gx = prod(extended_x, 1)
        gx = gx.reshape(transposed_shape)
        gx = gx.transpose(list(numpy.argsort(transpose_axes)))
        gy = chainer.functions.broadcast_to(gy, gx.shape)
        gx = gx * gy
        return gx,


def prod(x, axis=None, keepdims=False):
    """Product of array elements over a given axis.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Elements to calculate the product.
        axis (None, int, or tuple of int): Axis which a product is performed.
            The default (axis = None) is perform a product over all the
            dimensions of the input array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Prod(axis, keepdims).apply((x,))[0]
