import numpy
import six

import chainer
from chainer import backend
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
    xp = backend.get_array_module(indices)
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

    def __init__(self, indices, axis, inv):
        self.indices = indices
        self.axis = axis
        self.inv = inv

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        x_type, = in_types
        if self.axis < 0:
            type_check.expect(x_type.ndim >= -self.axis)
        else:
            type_check.expect(x_type.ndim > self.axis)

    def _permutate(self, x, indices, inv):
        if inv:
            indices = _inverse_indices(indices)

        return x[((slice(None),) * self.axis) + (indices,)]

    def forward(self, inputs):
        x, = inputs
        inds = self.indices

        if chainer.is_debug():
            _check_indices(inds)

        return self._permutate(x, inds, self.inv),

    def backward(self, indexes, grad_outputs):
        g, = grad_outputs
        inds = self.indices
        gx, = Permutate(inds, self.axis, not self.inv).apply((g,))
        return gx,


def permutate(x, indices, axis=0, inv=False):
    """Permutates a given variable along an axis.

    This function permutate ``x`` with given ``indices``.
    That means ``y[i] = x[indices[i]]`` for all ``i``.
    Note that this result is same as ``y = x.take(indices)``.
    ``indices`` must be a permutation of ``[0, 1, ..., len(x) - 1]``.

    When ``inv`` is ``True``, ``indices`` is treated as its inverse.
    That means ``y[indices[i]] = x[i]``.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Variable to permutate.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        indices (:class:`~chainer.Variable` or :ref:`ndarray`):
            Indices to extract from the variable. A one-dimensional int array.
        axis (int): Axis that the input array is permutate along.
        inv (bool): If ``True``, ``indices`` is treated as its inverse.

    Returns:
        ~chainer.Variable: Output variable.

    .. admonition:: Example

        >>> x = np.arange(6).reshape((3, 2)).astype(np.float32)
        >>> x
        array([[0., 1.],
               [2., 3.],
               [4., 5.]], dtype=float32)
        >>> indices = np.array([2, 0, 1], np.int32)
        >>> y = F.permutate(x, indices)
        >>> y.array
        array([[4., 5.],
               [0., 1.],
               [2., 3.]], dtype=float32)
        >>> y = F.permutate(x, indices, inv=True)
        >>> y.array
        array([[2., 3.],
               [4., 5.],
               [0., 1.]], dtype=float32)
        >>> indices = np.array([1, 0], np.int32)
        >>> y = F.permutate(x, indices, axis=1)
        >>> y.array
        array([[1., 0.],
               [3., 2.],
               [5., 4.]], dtype=float32)

    """
    if indices.dtype.kind != 'i' or indices.ndim != 1:
        raise ValueError(
            'indices should be a one-dimensional int array')
    if isinstance(indices, chainer.Variable):
        indices = indices.array
    y, = Permutate(indices, axis, inv).apply((x,))
    return y
