import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check
import chainerx


class ScatterAdd(function_node.FunctionNode):

    def __init__(self, slices):
        if isinstance(slices, list):
            if all([isinstance(s, int) for s in slices]):
                slices = slices,
            slices = tuple(slices)
        elif not isinstance(slices, tuple):
            slices = slices,

        if chainer.is_debug():
            n_ellipses = 0
            for s in slices:
                if s is Ellipsis:
                    n_ellipses += 1
            if n_ellipses > 1:
                raise ValueError('Only one Ellipsis is allowed')

        self.slices = slices

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('a', 'b'))
        n_nones = len([item for item in self.slices if item is None])
        valid_slice = len(self.slices) - n_nones
        type_check.expect(in_types[0].ndim >= valid_slice)

    def forward(self, xs):
        a = xs[0]
        b = xs[1]
        y = a.copy()
        xp = backend.get_array_module(a)
        slices = tuple([
            backend.from_chx(s) if isinstance(s, chainerx.ndarray) else s
            for s in self.slices])
        if y[slices].shape != b.shape:
            raise ValueError(
                'Chainer does not support automatic broadcasting '
                'of variables.')
        if xp is numpy:
            numpy.add.at(y, slices, b),
        else:
            cuda.cupyx.scatter_add(y, slices, b),
        return y,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs

        ret = []
        if 0 in indexes:
            ret.append(gy)
        if 1 in indexes:
            ret.append(gy[self.slices])

        return ret


def scatter_add(a, slices, b):
    """Adds given values to specified elements of an array.

    This function adds ``b`` to the specified elements of the copy of
    ``a``, and returns the copy.
    The value of the original ``a`` is not changed.

    Args:
        a (:class:`~chainer.Variable` or :ref:`ndarray`): A variable.
        slices (int, slice, Ellipsis, None, integer array-like, boolean\
        array-like or tuple of them):
            It is an integer, a slice, an ellipsis,
            a numpy.newaxis, an integer array-like, a boolean array-like
            or tuple of them.
        b (:class:`~chainer.Variable` or :ref:`ndarray`):
            A variable that is scatter added to ``a``.
            Its shape has to equal ``a[slices]`` because broadcasting
            of variables is not supported.

    Returns:
        A :class:`~chainer.Variable` object which is the result of
        scatter addition.

    .. note::

        It only supports types that are supported by CUDA's atomicAdd when
        an integer array is included in ``slices``.
        The supported types are ``numpy.float32``, ``numpy.int32``,
        ``numpy.uint32``, ``numpy.uint64`` and ``numpy.ulonglong``.

    .. note::

        It does not support ``slices`` that contains multiple boolean arrays.

    .. seealso::
        :func:`numpy.add.at` and
        :func:`cupyx.scatter_add`.

    """
    y, = ScatterAdd(slices).apply((a, b))
    return y
