import numpy
import six

from chainer import backend
from chainer import function_node
from chainer.functions.pooling import pooling_nd
from chainer.utils import conv
from chainer.utils import conv_nd
from chainer.utils import type_check


class UnpoolingND(pooling_nd._PoolingND):
    """Unpooling over a set of N-dimensional planes.

    .. warning::

        This feature is experimental. The interface can change in the future.

    """

    def __init__(self, ndim, ksize, stride=None, pad=0, outsize=None,
                 cover_all=True):
        super(UnpoolingND, self).__init__(ndim, ksize, stride, pad, cover_all)
        self.outs = None if outsize is None else outsize

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 1)
        x_type = in_types[0]

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 2 + self.ndim,
        )
        if self.outs is not None:
            expected_dims = tuple(
                conv.get_conv_outsize(out, k, s, p, cover_all=self.cover_all)
                for (out, k, s, p)
                in six.moves.zip(self.outs, self.ksize, self.stride, self.pad))
            type_check.expect(x_type.shape[2:] == expected_dims)

    def forward(self, x):
        self.retain_inputs(())

        dims = x[0].shape[2:]
        ndim = self.ndim
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        if self.outs is None:
            self.outs = tuple(
                conv.get_deconv_outsize(d, k, s, p, cover_all=self.cover_all)
                for (d, k, s, p) in six.moves.zip(dims, ksize, stride, pad))

        xp = backend.get_array_module(*x)

        colon = slice(None)
        # (:, :, None, None, ..., None)
        tile_index = (colon, colon) + (None,) * ndim
        # (1, 1, k_1, k_2, ..., k_n, 1, 1, ..., 1)
        tile_reps = (1, 1) + ksize + (1,) * ndim
        col = xp.tile(x[0][tile_index], tile_reps)

        if xp is numpy:
            col2im_nd = conv_nd.col2im_nd_cpu
        else:
            col2im_nd = conv_nd.col2im_nd_gpu
        y = col2im_nd(col, stride, pad, self.outs)

        return y,

    def backward(self, indexes, grad_outputs):
        return UnpoolingNDGrad(self).apply(grad_outputs)


class UnpoolingNDGrad(function_node.FunctionNode):

    def __init__(self, unpoolingnd):
        self.ndim = unpoolingnd.ndim
        self.ksize = unpoolingnd.ksize
        self.stride = unpoolingnd.stride
        self.pad = unpoolingnd.pad
        self.outs = unpoolingnd.outs
        self.cover_all = unpoolingnd.cover_all

    def forward(self, gy):
        xp = backend.get_array_module(*gy)
        if xp is numpy:
            im2col_nd = conv_nd.im2col_nd_cpu
        else:
            im2col_nd = conv_nd.im2col_nd_gpu
        gcol = im2col_nd(
            gy[0], self.ksize, self.stride, self.pad, cover_all=self.cover_all)
        gcol_axis = tuple(six.moves.range(2, 2 + self.ndim))
        gx = gcol.sum(axis=gcol_axis)
        return gx,

    def backward(self, indexes, ggx):
        return UnpoolingND(
            self.ndim, self.ksize, self.stride, self.pad, self.outs,
            self.cover_all).apply(ggx)


def unpooling_nd(x, ksize, stride=None, pad=0, outsize=None, cover_all=True):
    """Inverse operation of N-dimensional spatial pooling.

    .. warning::

        This feature is experimental. The interface can change in the future.

    This function acts similarly to
    :class:`~functions.connection.deconvolution_nd.DeconvolutionND`, but
    it spreads input N-dimensional array's value without any parameter instead
    of computing the inner products.

    Args:
        x (~chainer.Variable): Input variable.
        ksize (int or pair of ints): Size of pooling window
            :math:`(k_1, k_2, ..., k_N)`. ``ksize=k`` is equivalent to
            ``(k, k, ..., k)``.
        stride (int, pair of ints or None): Stride of pooling applications
            :math:`(s_1, s_2, ..., s_N)`. ``stride=s`` is equivalent to
            ``(s, s, ..., s)``. If ``None`` is specified, then it uses same
            stride as the pooling window size.
        pad (int or pair of ints): Spatial padding width for the input array
            :math:`(p_1, p_2, ..., p_N)`. ``pad=p`` is equivalent to
            ``(p, p, ..., p)``.
        outsize (None or pair of ints): Expected output size of unpooling
            operation :math:`(out_1, out_2, ..., out_N)`. If ``None``, the size
            is estimated from input size, stride and padding.
        cover_all (bool): If ``True``, the pooling window is assumed to cover
            all of the output array, eventually the output size may be smaller
            than that in the case ``cover_all`` is ``False``.

    Returns:
        ~chainer.Variable: Output variable.

    """
    ndim = len(x.shape[2:])
    return UnpoolingND(
        ndim, ksize, stride, pad, outsize, cover_all).apply((x,))[0]


def unpooling_1d(x, ksize, stride=None, pad=0, outsize=None, cover_all=True):
    """Inverse operation of 1-dimensional spatial pooling.

    .. warning::

        This feature is experimental. The interface can change in the future.

    .. note::

        This function calls :func:`~chainer.functions.unpooling_nd`
        internally, so see the details of the behavior in
        the documentation of :func:`~chainer.functions.unpooling_nd`.

    """
    if len(x.shape[2:]) != 1:
        raise ValueError(
            'The number of dimensions under channel dimension of the input '
            '\'x\' should be 1. But the actual ndim was {}.'.format(
                len(x.shape[2:])))
    return unpooling_nd(x, ksize, stride, pad, outsize, cover_all)


def unpooling_3d(x, ksize, stride=None, pad=0, outsize=None, cover_all=True):
    """Inverse operation of 3-dimensional spatial pooling.

    .. warning::

        This feature is experimental. The interface can change in the future.

    .. note::

        This function calls :func:`~chainer.functions.unpooling_nd`
        internally, so see the details of the behavior in
        the documentation of :func:`~chainer.functions.unpooling_nd`.

    """
    if len(x.shape[2:]) != 3:
        raise ValueError(
            'The number of dimensions under channel dimension of the input '
            '\'x\' should be 3. But the actual ndim was {}.'.format(
                len(x.shape[2:])))
    return unpooling_nd(x, ksize, stride, pad, outsize, cover_all)
