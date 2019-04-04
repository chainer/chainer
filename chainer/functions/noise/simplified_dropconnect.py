import numpy

from chainer import backend
from chainer import function_node
import chainer.functions
from chainer.utils import type_check
from chainer import variable


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


def _matmul(a, b, xp):
    if xp is numpy:
        # numpy 1.9 does not support matmul.
        # So we use numpy.einsum instead of numpy.matmul.
        return xp.einsum('...jk,...kl->...jl', a, b)
    else:
        return xp.matmul(a, b)


class SimplifiedDropconnect(function_node.FunctionNode):

    """Linear unit regularized by simplified dropconnect."""

    def __init__(self, ratio, mask=None, use_batchwise_mask=True):
        self.ratio = ratio
        self.mask = mask
        self.use_batchwise_mask = use_batchwise_mask

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]
        type_check._argname((x_type, w_type), ('x', 'W'))

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check._argname((b_type,), ('b',))
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

        if self.mask is not None:
            if self.use_batchwise_mask:
                type_check.expect(
                    self.mask.shape[0] == x_type.shape[0],
                    self.mask.shape[1:] == w_type.shape,
                )
            else:
                type_check.expect(self.mask.shape == w_type.shape)

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        scale = inputs[1].dtype.type(1. / (1 - self.ratio))
        xp = backend.get_array_module(*inputs)

        if self.mask is None:
            if self.use_batchwise_mask:
                mask_shape = (inputs[0].shape[0], inputs[1].shape[0],
                              inputs[1].shape[1])
            else:
                mask_shape = (inputs[1].shape[0], inputs[1].shape[1])
            if xp == numpy:
                self.mask = xp.random.rand(*mask_shape) >= self.ratio
            else:
                self.mask = xp.random.rand(*mask_shape,
                                           dtype=numpy.float32) >= self.ratio
        elif isinstance(self.mask, variable.Variable):
            self.mask = self.mask.data

        x = _as_mat(inputs[0])
        W = inputs[1] * scale * self.mask

        # (i)jk,ik->ij
        y = _matmul(W, x[:, :, None], xp)
        y = y.reshape(y.shape[0], y.shape[1]).astype(x.dtype, copy=False)

        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,

    def backward(self, indexes, grad_outputs):
        inputs = self.get_retained_inputs()
        ret = []

        scale = inputs[1].dtype.type(1. / (1 - self.ratio))
        x = _as_mat(inputs[0])

        W = inputs[1]
        if self.use_batchwise_mask:
            W = chainer.functions.broadcast_to(
                W, self.mask.shape) * scale * self.mask
        else:
            W = chainer.functions.broadcast_to(
                W * scale * self.mask, (x.shape[0],) + self.mask.shape)
        gy = grad_outputs[0]

        if 0 in indexes:
            # ij,(i)jk->ik
            gx = chainer.functions.matmul(
                gy[:, None, :], W).reshape(inputs[0].shape)
            gx = chainer.functions.cast(gx, x.dtype)
            ret.append(gx)

        if 1 in indexes:
            # ij,ik,ijk->jk
            gy2 = gy[:, :, None]
            x2 = x[:, None, :]
            shape = (gy2.shape[0], gy2.shape[1], x2.shape[2])
            gy2 = chainer.functions.broadcast_to(gy2, shape)
            x2 = chainer.functions.broadcast_to(x2, shape)
            gW = chainer.functions.sum(gy2 * x2 * self.mask, axis=0) * scale
            gW = chainer.functions.cast(gW, W.dtype)
            ret.append(gW)

        if 2 in indexes:
            gb = chainer.functions.sum(gy, axis=0)
            ret.append(gb)

        return ret


def simplified_dropconnect(x, W, b=None, ratio=.5, train=True, mask=None,
                           use_batchwise_mask=True):
    """Linear unit regularized by simplified dropconnect.

    Simplified dropconnect drops weight matrix elements randomly with
    probability ``ratio`` and scales the remaining elements by factor
    ``1 / (1 - ratio)``.
    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes
    :math:`Y = xW^\\top + b`.

    In testing mode, zero will be used as simplified dropconnect ratio instead
    of ``ratio``.

    Notice:
    This implementation cannot be used for reproduction of the paper.
    There is a difference between the current implementation and the
    original one.
    The original version uses sampling with gaussian distribution before
    passing activation function, whereas the current implementation averages
    before activation.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable. Its first dimension ``n`` is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as concatenated one dimension whose size must be ``N``.
        W (:class:`~chainer.Variable` or :ref:`ndarray`):
            Weight variable of shape ``(M, N)``.
        b (:class:`~chainer.Variable` or :ref:`ndarray`):
            Bias variable (optional) of shape ``(M,)``.
        ratio (float):
            Dropconnect ratio.
        train (bool):
            If ``True``, executes simplified dropconnect.
            Otherwise, simplified dropconnect function works as a linear
            function.
        mask (None or :class:`~chainer.Variable` or :ref:`ndarray`):
            If ``None``, randomized dropconnect mask is generated.
            Otherwise, The mask must be ``(n, M, N)`` or ``(M, N)`` shaped
            array, and `use_batchwise_mask` is ignored.
            Main purpose of this option is debugging.
            `mask` array will be used as a dropconnect mask.
        use_batchwise_mask (bool):
            If ``True``, dropped connections depend on each sample in
            mini-batch.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`~chainer.links.Dropconnect`

    .. seealso::
        Li, W., Matthew Z., Sixin Z., Yann L., Rob F. (2013).
        Regularization of Neural Network using DropConnect.
        International Conference on Machine Learning.
        `URL <https://cs.nyu.edu/~wanli/dropc/>`_
    """
    if not train:
        ratio = 0
    if b is None:
        return SimplifiedDropconnect(
            ratio, mask, use_batchwise_mask).apply((x, W))[0]
    else:
        return SimplifiedDropconnect(
            ratio, mask, use_batchwise_mask).apply((x, W, b))[0]
