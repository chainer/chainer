import numpy
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


class SelectItem(function_node.FunctionNode):

    """Select elements stored in given indices."""

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x', 't'))

        x_type, t_type = in_types
        type_check.expect(
            t_type.dtype.kind == 'i',
            x_type.ndim == 2,
            t_type.ndim == 1,
            x_type.shape[0] == t_type.shape[0],
        )

    def forward(self, inputs):
        self.retain_inputs((1,))
        x, t = inputs
        self._in_shape = x.shape
        self._in_dtype = x.dtype
        if chainer.is_debug():
            if not ((0 <= t).all() and
                    (t < x.shape[1]).all()):
                msg = 'Each label `t` need to satisfty `0 <= t < x.shape[1]`'
                raise ValueError(msg)

        xp = backend.get_array_module(x)
        if xp is numpy:
            # This code is equivalent to `t.choose(x.T)`, but `numpy.choose`
            # does not work when `x.shape[1] > 32`.
            return x[six.moves.range(t.size), t],
        else:
            y = cuda.elementwise(
                'S t, raw T x',
                'T y',
                'int ind[] = {i, t}; y = x[ind];',
                'getitem_fwd'
            )(t, x)
            return y,

    def backward(self, indexes, gy):
        t = self.get_retained_inputs()[0]
        ret = []
        if 0 in indexes:
            gx = Assign(self._in_shape, self._in_dtype, t).apply(gy)[0]
            ret.append(gx)
        if 1 in indexes:
            ret.append(None)
        return ret


class Assign(function_node.FunctionNode):

    def __init__(self, shape, dtype, t):
        self.shape = shape
        self.dtype = dtype
        self.t = t.data

    def forward_cpu(self, inputs):
        t = backend.from_chx(self.t)  # Workaround for ChainerX.

        gx = numpy.zeros(self.shape, self.dtype)
        gx[six.moves.range(self.t.size), t] = inputs[0]
        return gx,

    def forward_gpu(self, inputs):
        t = backend.from_chx(self.t)  # Workaround for ChainerX.

        gx = cuda.cupy.zeros(self.shape, self.dtype)
        gx = cuda.elementwise(
            'S t, T gloss',
            'raw T gx',
            'int ind[] = {i, t}; gx[ind] = gloss;',
            'getitem_bwd'
        )(t, inputs[0], gx)
        return gx,

    def backward(self, indexes, gy):
        return SelectItem().apply((gy[0], self.t))


def select_item(x, t):
    """Select elements stored in given indices.

    This function returns ``t.choose(x.T)``, that means
    ``y[i] == x[i, t[i]]`` for all ``i``.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Variable storing arrays. A two-dimensional float array.
        t (:class:`~chainer.Variable` or :ref:`ndarray`):
            Variable storing index numbers. A one-dimensional int array.
            Length of the ``t`` should be equal to ``x.shape[0]``.

    Returns:
        ~chainer.Variable: Variable that holds ``t``-th element of ``x``.

    .. admonition:: Example

        >>> x = np.array([[0, 1, 2], [3, 4, 5]], np.float32)
        >>> t = np.array([0, 2], np.int32)
        >>> y = F.select_item(x, t)
        >>> y.shape
        (2,)
        >>> y.array
        array([0., 5.], dtype=float32)

    """
    return SelectItem().apply((x, t))[0]
