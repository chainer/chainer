import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


def _check_indices(indices):
    if len(indices) == 0:
        return
    # TODO(unno): Check indices without cpu
    indices = cuda.to_cpu(indices)
    for i in indices:
        if 0 <= i < len(indices):
            continue
        raise ValueError('Out of bounds index: {}'.format(i))
    sort = numpy.sort(indices)
    for s, t in six.moves.zip(sort, sort[1:]):
        if s == t:
            raise ValueError('indices contains duplicate value: {}'.format(s))


def _inverse_indices(indices):
    xp = cuda.get_array_module(indices)
    r = xp.empty_like(indices)
    if xp is numpy:
        r[indices] = numpy.arange(len(indices))
    else:
        cuda.elementwise(
            'S ind', 'raw S r',
            'r[ind] = i',
            'inverse_indices'
        )(indices, r)
    return r


class Permutate(function_node.FunctionNode):

    """Permutate function."""

    def __init__(self, axis=0, inv=False):
        self.axis = axis
        self.inv = inv

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, ind_type = in_types
        if self.axis < 0:
            type_check.expect(x_type.ndim >= -self.axis)
        else:
            type_check.expect(x_type.ndim > self.axis)

        type_check.expect(
            ind_type.dtype.kind == 'i',
            ind_type.ndim == 1,
            x_type.shape[self.axis] == ind_type.shape[0],
        )

    def _permutate(self, x, indices, inv):
        if inv:
            indices = _inverse_indices(indices)

        return x[((slice(None),) * self.axis) + (indices,)]

    def forward(self, inputs):
        self.retain_inputs((1,))
        x, inds = inputs

        if chainer.is_debug():
            _check_indices(inds)

        return self._permutate(x, inds, self.inv),

    def backward(self, indexes, grad_outputs):
        inds = self.inputs[1]
        g, = grad_outputs
        gx, = Permutate(self.axis, not self.inv).apply((g, inds.data))
        return gx, None


def permutate(x, indices, axis=0, inv=False):
    """Permutates a given variable along an axis.

    This function permutate ``x`` with given ``indices``.
    That means ``y[i] = x[indices[i]]`` for all ``i``.
    Note that this result is same as ``y = x.take(indices)``.
    ``indices`` must be a permutation of ``[0, 1, ..., len(x) - 1]``.

    When ``inv`` is ``True``, ``indices`` is treated as its inverse.
    That means ``y[indices[i]] = x[i]``.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable to permutate.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        indices (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Indices to extract from the variable. A one-dimensional int array.
        axis (int): Axis that the input array is permutate along.
        inv (bool): If ``True``, ``indices`` is treated as its inverse.

    Returns:
        ~chainer.Variable: Output variable.

    .. admonition:: Example

        >>> x = np.arange(6).reshape((3, 2)).astype('f')
        >>> x
        array([[0., 1.],
               [2., 3.],
               [4., 5.]], dtype=float32)
        >>> indices = np.array([2, 0, 1], 'i')
        >>> y = F.permutate(x, indices)
        >>> y.data
        array([[4., 5.],
               [0., 1.],
               [2., 3.]], dtype=float32)
        >>> y = F.permutate(x, indices, inv=True)
        >>> y.data
        array([[2., 3.],
               [4., 5.],
               [0., 1.]], dtype=float32)
        >>> indices = np.array([1, 0], 'i')
        >>> y = F.permutate(x, indices, axis=1)
        >>> y.data
        array([[1., 0.],
               [3., 2.],
               [5., 4.]], dtype=float32)

    """
    y, = Permutate(axis, inv).apply((x, indices))
    return y
