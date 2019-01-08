import numpy

import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check


class Depth2Space(function_node.FunctionNode):

    """Depth to space transformation."""

    def __init__(self, r):
        self.r = r

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim == 4
        )

    def forward(self, inputs):
        X, = inputs
        xp = backend.get_array_module(X)
        bsize, c, a, b = X.shape
        c //= self.r ** 2

        if xp is numpy:
            # These codes run faster on CPU than below `else` block codes.
            X = xp.transpose(X, (0, 2, 3, 1))
            X = xp.reshape(X, (bsize, a, b, self.r, self.r, c))
            X = xp.transpose(X, (0, 1, 3, 2, 4, 5))
            X = xp.reshape(X, (bsize, a * self.r, b * self.r, c))
            X = xp.transpose(X, (0, 3, 1, 2))
        else:
            X = xp.reshape(X, (bsize, self.r, self.r, c, a, b))
            X = xp.transpose(X, (0, 3, 4, 1, 5, 2))
            X = xp.reshape(X, (bsize, c, a * self.r, b * self.r))
        return X,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        gy = chainer.functions.space2depth(gy, self.r)
        return gy,


def depth2space(X, r):
    """Computes the depth2space transformation for subpixel calculations.

    Args:
        X (:class:`~chainer.Variable` or :ref:`ndarray`): Variable holding a
            4d array of shape ``(batch, channel * r * r, dim1, dim2)``.
        r (int): the upscaling factor.

    Returns:
        ~chainer.Variable:
            A variable holding the upscaled array from
            interspersed depth layers. The shape is
            ``(batch, channel, dim1 * r, dim2 * r)``.

    .. note::
       This can be used to compute super-resolution transformations.
       See https://arxiv.org/abs/1609.05158 for details.

    .. seealso:: :func:`space2depth`

    .. admonition:: Example

        >>> X = np.arange(24).reshape(1, 4, 2, 3).astype(np.float32)
        >>> X.shape
        (1, 4, 2, 3)
        >>> X
        array([[[[ 0.,  1.,  2.],
                 [ 3.,  4.,  5.]],
        <BLANKLINE>
                [[ 6.,  7.,  8.],
                 [ 9., 10., 11.]],
        <BLANKLINE>
                [[12., 13., 14.],
                 [15., 16., 17.]],
        <BLANKLINE>
                [[18., 19., 20.],
                 [21., 22., 23.]]]], dtype=float32)
        >>> y = F.depth2space(X, 2)
        >>> y.shape
        (1, 1, 4, 6)
        >>> y.array
        array([[[[ 0.,  6.,  1.,  7.,  2.,  8.],
                 [12., 18., 13., 19., 14., 20.],
                 [ 3.,  9.,  4., 10.,  5., 11.],
                 [15., 21., 16., 22., 17., 23.]]]], dtype=float32)

    """
    return Depth2Space(r).apply((X,))[0]
