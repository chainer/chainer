import numpy
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class EmbedIDFunction(function_node.FunctionNode):

    def __init__(self, ignore_label=None):
        self.ignore_label = ignore_label

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, w_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'i',
            x_type.ndim >= 1,
        )
        type_check.expect(
            w_type.dtype.kind == 'f',
            w_type.ndim == 2
        )

    def forward(self, inputs):
        self.retain_inputs((0,))
        x, W = inputs
        self._w_shape = W.shape

        xp = backend.get_array_module(*inputs)
        if chainer.is_debug():
            valid_x = xp.logical_and(0 <= x, x < len(W))
            if self.ignore_label is not None:
                valid_x = xp.logical_or(valid_x, x == self.ignore_label)
            if not valid_x.all():
                raise ValueError('Each not ignored `x` value need to satisfy '
                                 '`0 <= x < len(W)`')

        if self.ignore_label is not None:
            mask = (x == self.ignore_label)
            return xp.where(mask[..., None], 0, W[xp.where(mask, 0, x)]),

        return W[x],

    def backward(self, indexes, grad_outputs):
        inputs = self.get_retained_inputs()
        gW = EmbedIDGrad(
            self._w_shape, self.ignore_label).apply(inputs + grad_outputs)[0]
        return None, gW


class EmbedIDGrad(function_node.FunctionNode):

    def __init__(self, w_shape, ignore_label=None):
        self.w_shape = w_shape
        self.ignore_label = ignore_label

    def forward(self, inputs):
        self.retain_inputs((0,))
        xp = backend.get_array_module(*inputs)
        x, gy = inputs
        self._gy_shape = gy.shape
        gW = xp.zeros(self.w_shape, dtype=gy.dtype)

        if xp is numpy:
            # It is equivalent to `numpy.add.at(gW, x, gy)` but ufunc.at is
            # too slow.
            for ix, igy in six.moves.zip(x.ravel(),
                                         gy.reshape(x.size, -1)):
                if ix == self.ignore_label:
                    continue
                gW[ix] += igy
        else:
            utils.nondeterministic('atomicAdd')
            if self.ignore_label is None:
                cuda.elementwise(
                    'T gy, S x, S n_out', 'raw T gW',
                    'ptrdiff_t w_ind[] = {x, i % n_out};'
                    'atomicAdd(&gW[w_ind], gy)',
                    'embed_id_bwd')(
                        gy, xp.expand_dims(x, -1), gW.shape[1], gW)
            else:
                cuda.elementwise(
                    'T gy, S x, S n_out, S ignore', 'raw T gW',
                    '''
                    if (x != ignore) {
                      ptrdiff_t w_ind[] = {x, i % n_out};
                      atomicAdd(&gW[w_ind], gy);
                    }
                    ''',
                    'embed_id_bwd_ignore_label')(
                        gy, xp.expand_dims(x, -1), gW.shape[1],
                        self.ignore_label, gW)
        return gW,

    def backward(self, indexes, grads):
        xp = backend.get_array_module(*grads)
        x = self.get_retained_inputs()[0].data
        ggW = grads[0]

        if self.ignore_label is not None:
            mask = x == self.ignore_label
            # To prevent index out of bounds, we need to check if ignore_label
            # is inside of W.
            if not (0 <= self.ignore_label < self.w_shape[1]):
                x = xp.where(mask, 0, x)

        ggy = ggW[x]

        if self.ignore_label is not None:
            mask, zero, _ = xp.broadcast_arrays(
                mask[..., None], xp.zeros((), ggy.dtype), ggy.data)
            ggy = chainer.functions.where(mask, zero, ggy)
        return None, ggy


def embed_id(x, W, ignore_label=None):
    """Efficient linear function for one-hot input.

    This function implements so called *word embeddings*. It takes two
    arguments: a set of IDs (words) ``x`` in :math:`B` dimensional integer
    vector, and a set of all ID (word) embeddings ``W`` in :math:`V \\times d`
    float matrix. It outputs :math:`B \\times d` matrix whose ``i``-th
    row is the ``x[i]``-th row of ``W``.

    This function is only differentiable on the input ``W``.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Batch vectors of IDs. Each element must be signed integer.
        W (:class:`~chainer.Variable` or :ref:`ndarray`):
            Distributed representation of each ID (a.k.a. word embeddings).
        ignore_label (:class:`int` or :class:`None`):
            If ``ignore_label`` is an int value, ``i``-th row of return
            value is filled with ``0``.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso::

        :class:`~chainer.links.EmbedID` to manage the model parameter ``W``.

    .. admonition:: Example

        >>> x = np.array([2, 1]).astype(np.int32)
        >>> x
        array([2, 1], dtype=int32)
        >>> W = np.array([[0, 0, 0],
        ...               [1, 1, 1],
        ...               [2, 2, 2]]).astype(np.float32)
        >>> W
        array([[0., 0., 0.],
               [1., 1., 1.],
               [2., 2., 2.]], dtype=float32)
        >>> F.embed_id(x, W).array
        array([[2., 2., 2.],
               [1., 1., 1.]], dtype=float32)
        >>> F.embed_id(x, W, ignore_label=1).array
        array([[2., 2., 2.],
               [0., 0., 0.]], dtype=float32)

    """
    return EmbedIDFunction(ignore_label=ignore_label).apply((x, W))[0]
