import numpy

from chainer import backend
from chainer import function_node
import chainer.functions
import chainer.utils
from chainer.utils import type_check
import chainerx


class SelectorBase(function_node.FunctionNode):
    """Select an array element from a given axis or set of axes."""

    def __init__(self, axis=None, keepdims=False):
        self.keepdims = keepdims
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

    def _fwd(self, x, xp):
        raise NotImplementedError('_fwd should be implemented in sub-class.')

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
        self.retain_outputs((0,))
        xp = backend.get_array_module(*x)
        return xp.asarray(self._fwd(x[0], xp)),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()[0]
        y = self.get_retained_outputs()[0]

        if self.axis is None:
            axis = range(x.ndim)
        else:
            axis = [ax % x.ndim for ax in self.axis]

        # Add broadcastable dimensions to y and gy
        # for each one that was reduced in the forward operation
        shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
        gy = gy[0].reshape(shape)
        y = y.reshape(shape)

        # Compute the gradient
        cond = (x.data == y.data)
        gy = chainer.functions.broadcast_to(gy, cond.shape)
        return gy * cond,


class Max(SelectorBase):

    def forward_chainerx(self, x):
        return chainerx.amax(x[0], axis=self.axis, keepdims=self.keepdims),

    def _fwd(self, x, xp):
        return xp.amax(x, axis=self.axis, keepdims=self.keepdims)


class Min(SelectorBase):

    def forward_chainerx(self, x):
        return chainerx.amin(x[0], axis=self.axis, keepdims=self.keepdims),

    def _fwd(self, x, xp):
        return xp.amin(x, axis=self.axis, keepdims=self.keepdims)


class IndexSelectorBase(function_node.FunctionNode):
    """Select index of an array element from a given axis."""

    def __init__(self, axis=None):
        if axis is None:
            self.axis = None
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError('None or int are required')

    def _fwd(self, x, xp):
        raise NotImplementedError('_fwd should be implemented in sub-class.')

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f'
        )

        if self.axis is not None:
            if self.axis >= 0:
                type_check.expect(
                    self.axis < in_types[0].ndim,
                )
            else:
                type_check.expect(
                    -self.axis - 1 < in_types[0].ndim,
                )

    def forward(self, x):
        xp = backend.get_array_module(*x)
        return xp.asarray(self._fwd(x[0], xp)),

    def backward(self, indexes, grad_outputs):
        return None,


class ArgMin(IndexSelectorBase):

    def _fwd(self, x, xp):
        return xp.argmin(x, axis=self.axis).astype(numpy.int32)


class ArgMax(IndexSelectorBase):

    def forward_chainerx(self, x):
        return chainerx.argmax(x[0], axis=self.axis).astype(numpy.int32),

    def _fwd(self, x, xp):
        return xp.argmax(x, axis=self.axis).astype(numpy.int32)


def max(x, axis=None, keepdims=False):
    """Maximum of array elements over a given axis.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Array to be maximized.
        axis (None, int, or tuple of int): Axis over which a max is performed.
            The default (axis = None) is perform a max over all the dimensions
            of the input array.
    Returns:
        ~chainer.Variable: Output variable.

    """
    return Max(axis, keepdims).apply((x,))[0]


def min(x, axis=None, keepdims=False):
    """Minimum of array elements over a given axis.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Array to be minimized.
        axis (None, int, or tuple of int): Axis over which a min is performed.
            The default (axis = None) is perform a min over all the dimensions
            of the input array.
    Returns:
        ~chainer.Variable: Output variable.

    """
    return Min(axis, keepdims).apply((x,))[0]


def argmax(x, axis=None):
    """Returns index which holds maximum of array elements over a given axis.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Array to find maximum elements.
        axis (None or int): Axis over which a max is performed.
            The default (axis = None) is perform a max over all the dimensions
            of the input array.
    Returns:
        ~chainer.Variable: Output variable.

    """
    return ArgMax(axis).apply((x,))[0]


def argmin(x, axis=None):
    """Returns index which holds minimum of array elements over a given axis.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Array to find minimum elements.
        axis (None or int): Axis over which a min is performed.
            The default (axis = None) is perform a min over all the dimensions
            of the input array.
    Returns:
        ~chainer.Variable: Output variable.

    """
    return ArgMin(axis).apply((x,))[0]
