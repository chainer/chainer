import numpy
from six import moves

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import configuration
from chainer import function_node
from chainer.functions.connection import convolution_2d
from chainer import utils
from chainer.utils import conv
from chainer.utils import conv_nd
from chainer.utils import type_check
import chainerx


class ConvolutionND(function_node.FunctionNode):

    def __init__(self, ndim, stride=1, pad=0, cover_all=False,
                 dilate=1, groups=1):
        self.ndim = ndim
        self.stride = conv_nd.as_tuple(stride, ndim)
        self.pad = conv_nd.as_tuple(pad, ndim)
        self.cover_all = cover_all
        self.dilate = conv_nd.as_tuple(dilate, ndim)
        self.groups = groups

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == self.ndim + 2,
            w_type.ndim == self.ndim + 2,
            # Need to consider the case that group count > 1.
            # x_type.shape[1] == w_type.shape[1],
        )

        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype.kind == 'f',
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward_chainerx(self, inputs):
        # TODO(hvy): Support mixed precision.
        if any([arr.dtype != inputs[0].dtype for arr in inputs[1:]]):
            return chainer.Fallback
        # TODO(hvy): Support dilate > 1.
        if any(d != 1 for d in self.dilate):
            return chainer.Fallback
        # TODO(hvy): Support groups > 1.
        if self.groups > 1:
            return chainer.Fallback
        if inputs[0].device.backend.name == 'cuda' and (
                self.cover_all or self.ndim < 2):
            return chainer.Fallback

        return chainerx.conv(
            *inputs, stride=self.stride, pad=self.pad,
            cover_all=self.cover_all),

    def _use_cudnn(self, x, W):
        if cuda._cudnn_version < 6000 and any(d != 1 for d in self.dilate):
            # cuDNN < 6.0 does not support dilated convolutions
            return False
        if cuda._cudnn_version < 7000 and 1 < self.groups:
            # cuDNN < 7.0 does not support grouped convolutions
            return False
        return (
            chainer.should_use_cudnn('>=auto')
            and not self.cover_all
            and x.dtype == W.dtype
            and self.ndim > 1)

    def _forward_xp(self, x, W, b, xp):
        if 1 < self.groups:
            return self._forward_grouped_convolution_xp(x, W, b, xp)
        else:
            return self._forward_xp_core(x, W, b, xp)

    def _forward_grouped_convolution_xp(self, x, W, b, xp):
        # G: group count
        # N: batch size
        # iC: input channels
        # oC: output channels
        G = self.groups
        N, iC = x.shape[:2]
        oC = W.shape[0]
        k_size = W.shape[2:]
        iCg = iC // G
        oCg = oC // G
        dims = len(k_size)
        if iC % G != 0:
            raise TypeError('The number of groups must be '
                            'a divisor of that of input channels')
        if oC % G != 0:
            raise TypeError('The number of groups must be '
                            'a divisor of that of output channels')

        xp = backend.get_array_module(x)

        # (N, iC, k_size..., o_size...)
        x = conv_nd.im2col_nd(x, k_size, self.stride, self.pad,
                              cover_all=self.cover_all, dilate=self.dilate)
        o_size = x.shape[-dims:]

        x = xp.rollaxis(x, 0, dims + 2)  # (iC, k_size..., N, o_size...)
        mul_len = iCg * utils.size_of_shape(k_size)
        x = x.reshape(G, mul_len, N * utils.size_of_shape(o_size))

        W = W.reshape(G, oCg, mul_len)

        # (G, oCg, N*o_size) = (G, oCg, iCg*k_size) @ (G, iCg*k_size, N*o_size)
        y = convolution_2d._matmul(W, x).astype(x.dtype, copy=False)
        y = y.reshape(oC, N, *o_size)
        y = xp.rollaxis(y, 1)  # (N, oC, o_size...)
        if b is not None:
            y += b.reshape(1, b.size, *((1,) * dims))

        return y,

    def _forward_xp_core(self, x, W, b, xp):
        ndim = self.ndim
        ksize = W.shape[2:]
        stride = self.stride
        pad = self.pad
        dilate = self.dilate

        # Make patch array.
        if xp is numpy:
            col = conv_nd.im2col_nd_cpu(
                x, ksize, stride, pad, cover_all=self.cover_all, dilate=dilate)
        else:
            col = conv_nd.im2col_nd_gpu(
                x, ksize, stride, pad, cover_all=self.cover_all, dilate=dilate)

        # Compute correlation.
        axes = tuple(moves.range(1, ndim + 2))  # (1, 2, ..., N+1)
        y = xp.tensordot(col, W, (axes, axes)).astype(x.dtype, copy=False)

        # Apply bias if given.
        if b is not None:
            y += b

        # Roll c_O before the second in (n, y_1, y_2, ..., y_N, c_O).
        return xp.rollaxis(y, ndim + 1, 1),

    def _forward_cudnn(self, x, W, b):
        out_c = W.shape[0]      # (c_O, _, k_1, k_2, ..., k_N)
        ksize = W.shape[2:]
        n, c = x.shape[:2]      # (n, c_I, d_1, d_2, ..., d_N)
        dims = x.shape[2:]
        stride = self.stride
        pad = self.pad
        dilate = self.dilate
        groups = self.groups

        # Make empty array for result.
        outs = tuple(
            conv.get_conv_outsize(d, k, s, p, cover_all=self.cover_all, d=di)
            for (d, k, s, p, di) in zip(dims, ksize, stride, pad, dilate))
        assert all(out > 0 for out in outs), 'Output sizes should be positive.'
        y_shape = (n, out_c) + outs  # (n, c_O, out_1, out_2, ..., out_N)
        y = cuda.cupy.empty(y_shape, dtype=x.dtype)
        auto_tune = configuration.config.autotune
        tensor_core = configuration.config.use_cudnn_tensor_core
        cuda.cudnn.convolution_forward(
            x, W, b, y, pad, stride, dilate, groups,
            auto_tune=auto_tune, tensor_core=tensor_core)
        return y,

    def forward(self, inputs):
        self.retain_inputs((0, 1))  # retain only x and W
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None

        xp = backend.get_array_module(*inputs)
        if xp is numpy:
            return self._forward_xp(x, W, b, numpy)
        elif not self._use_cudnn(x, W):
            return self._forward_xp(x, W, b, cuda.cupy)
        else:
            return self._forward_cudnn(x, W, b)

    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        gy, = grad_outputs

        ret = []
        if 0 in indexes:
            x_shape = x.shape[2:]
            gx = chainer.functions.deconvolution_nd(
                gy, W, stride=self.stride, pad=self.pad, outsize=x_shape,
                dilate=self.dilate, groups=self.groups)
            ret.append(gx)
        if 1 in indexes:
            gW, = ConvolutionNDGradW(self).apply((x, gy))
            ret.append(gW)
        if 2 in indexes:
            axis = (0,) + tuple(moves.range(2, gy.ndim))
            gb = chainer.functions.sum(gy, axis=axis)
            if gb.dtype != self.inputs[2].dtype:
                gb = chainer.functions.cast(gb, self.inputs[2].dtype)
            ret.append(gb)

        return ret


class ConvolutionNDGradW(function_node.FunctionNode):

    def __init__(self, convnd):
        W_node = convnd.inputs[1]
        self.ndim = convnd.ndim
        self.ksize = W_node.shape[2:]
        self.stride = convnd.stride
        self.pad = convnd.pad
        self.cover_all = convnd.cover_all
        self.dilate = convnd.dilate
        self.groups = convnd.groups
        self.W_dtype = W_node.dtype

    def _use_cudnn(self, x, gy):
        if cuda._cudnn_version < 6000 and any(d != 1 for d in self.dilate):
            # cuDNN < 6.0 does not support dilated convolutions
            return False
        if cuda._cudnn_version < 7000 and 1 < self.groups:
            # cuDNN < 7.0 does not support grouped convolutions
            return False
        return (
            chainer.should_use_cudnn('>=auto')
            and not self.cover_all
            and x.dtype == self.W_dtype
            and gy.dtype == self.W_dtype
            and self.ndim > 1)

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs

        xp = backend.get_array_module(*inputs)
        if xp is numpy:
            return self._forward_xp(x, gy, numpy)
        elif not self._use_cudnn(x, gy):
            return self._forward_xp(x, gy, cuda.cupy)
        else:
            return self._forward_cudnn(x, gy)

    def _forward_xp(self, x, gy, xp):
        if 1 < self.groups:
            return self._forward_grouped_convolution_xp(x, gy, xp)
        else:
            return self._forward_xp_core(x, gy, xp)

    def _forward_grouped_convolution_xp(self, x, gy, xp):
        G = self.groups
        N, iC = x.shape[:2]
        oC = gy.shape[1]
        o_size = gy.shape[2:]
        o_size_prod = utils.size_of_shape(o_size)
        k_size = self.ksize
        dims = len(o_size)
        iCg = iC // G
        oCg = oC // G

        # Do not check iCg and oCg because this class is rarely used alone

        # (N, iC, k_size..., o_size...)
        x = conv_nd.im2col_nd(x, k_size, self.stride, self.pad,
                              cover_all=self.cover_all, dilate=self.dilate)

        x = xp.rollaxis(x, 0, dims + 2)  # (iC, k_size..., N, o_size...)
        mul_len = iCg * utils.size_of_shape(k_size)
        x = x.reshape(G, mul_len, N * o_size_prod)
        x = x.transpose(0, 2, 1)  # (G, N*o_size, iCg*k_size)

        gy = xp.rollaxis(gy, 1)  # (oC, N, o_size...)
        gy = gy.reshape(G, oCg, N * o_size_prod)

        # (G, oCg, iCg*k_size) = (G, oCg, N*o_size) @ (G, N*o_size, iCg*k_size)
        gW = convolution_2d._matmul(gy, x).astype(self.W_dtype, copy=False)
        gW = gW.reshape(oC, iCg, *k_size)

        return gW,

    def _forward_xp_core(self, x, gy, xp):
        # Compute filter weight gradient.
        # (n, _, out_1, out_2, ..., out_N)
        out_axes = (0,) + tuple(moves.range(2, self.ndim + 2))
        # (n, _, _, ..., _, out_1, out_2, ..., out_N)
        col_axes = (0,) + tuple(moves.range(self.ndim + 2, self.ndim * 2 + 2))

        # NumPy raises an error when the array is not contiguous.
        # See: https://github.com/chainer/chainer/issues/2744
        # TODO(niboshi): Remove this code when NumPy is fixed.
        if (xp is numpy and
                not (gy.flags.c_contiguous or gy.flags.f_contiguous) and
                1 in gy.shape):
            gy = numpy.ascontiguousarray(gy)

        if xp is numpy:
            col = conv_nd.im2col_nd_cpu(
                x, self.ksize, self.stride, self.pad,
                cover_all=self.cover_all, dilate=self.dilate)
        else:
            col = conv_nd.im2col_nd_gpu(
                x, self.ksize, self.stride, self.pad,
                cover_all=self.cover_all, dilate=self.dilate)
        gW = xp.tensordot(gy, col, (out_axes, col_axes)).astype(
            self.W_dtype, copy=False)
        return gW,

    def _forward_cudnn(self, x, gy):
        # Make empty arrays for result.
        out_c = gy.shape[1]
        in_c = x.shape[1] // self.groups
        gW = cuda.cupy.empty(
            (out_c, in_c) + self.ksize, dtype=self.W_dtype)

        # Compute
        pad = self.pad
        stride = self.stride
        dilate = self.dilate
        groups = self.groups
        deterministic = configuration.config.cudnn_deterministic
        auto_tune = configuration.config.autotune
        tensor_core = configuration.config.use_cudnn_tensor_core
        cuda.cudnn.convolution_backward_filter(
            x, gy, gW, pad, stride, dilate, groups,
            deterministic=deterministic, auto_tune=auto_tune,
            tensor_core=tensor_core)

        return gW,

    def backward(self, indexes, grad_outputs):
        x, gy = self.get_retained_inputs()
        ggW, = grad_outputs

        ret = []
        if 0 in indexes:
            x_shape = x.shape[2:]
            gx = chainer.functions.deconvolution_nd(
                gy, ggW, stride=self.stride, pad=self.pad, outsize=x_shape,
                groups=self.groups, dilate=self.dilate)
            ret.append(gx)
        if 1 in indexes:
            ggy = convolution_nd(
                x, ggW, stride=self.stride, pad=self.pad,
                cover_all=self.cover_all, groups=self.groups,
                dilate=self.dilate)
            ret.append(ggy)

        return ret


def convolution_nd(x, W, b=None, stride=1, pad=0, cover_all=False,
                   dilate=1, groups=1):
    """N-dimensional convolution function.

    This is an implementation of N-dimensional convolution which is generalized
    two-dimensional convolution in ConvNets. It takes three variables: the
    input ``x``, the filter weight ``W`` and the bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`N` is the number of spatial dimensions.
    - :math:`n` is the batch size.
    - :math:`c_I` and :math:`c_O` are the number of the input and output
      channels, respectively.
    - :math:`d_1, d_2, ..., d_N` are the size of each axis of the input's
      spatial dimensions, respectively.
    - :math:`k_1, k_2, ..., k_N` are the size of each axis of the filters,
      respectively.
    - :math:`l_1, l_2, ..., l_N` are the size of each axis of the output's
      spatial dimensions, respectively.
    - :math:`p_1, p_2, ..., p_N` are the size of each axis of the spatial
      padding size, respectively.

    Then the ``convolution_nd`` function computes correlations between filters
    and patches of size :math:`(k_1, k_2, ..., k_N)` in ``x``.
    Note that correlation here is equivalent to the inner product between
    expanded tensors.
    Patches are extracted at positions shifted by multiples of ``stride`` from
    the first position ``(-p_1, -p_2, ..., -p_N)`` for each spatial axis.

    Let :math:`(s_1, s_2, ..., s_N)` be the stride of filter application.
    Then, the output size :math:`(l_1, l_2, ..., l_N)` is determined by the
    following equations:

    .. math::

       l_n = (d_n + 2p_n - k_n) / s_n + 1 \\ \\ (n = 1, ..., N)

    If ``cover_all`` option is ``True``, the filter will cover the all
    spatial locations. So, if the last stride of filter does not cover the
    end of spatial locations, an additional stride will be applied to the end
    part of spatial locations. In this case, the output size is determined by
    the following equations:

    .. math::

       l_n = (d_n + 2p_n - k_n + s_n - 1) / s_n + 1 \\ \\ (n = 1, ..., N)

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable of shape :math:`(n, c_I, d_1, d_2, ..., d_N)`.
        W (:class:`~chainer.Variable` or :ref:`ndarray`):
            Weight variable of shape :math:`(c_O, c_I, k_1, k_2, ..., k_N)`.
        b (None or :class:`~chainer.Variable` or :ref:`ndarray`):
            One-dimensional bias variable with length :math:`c_O` (optional).
        stride (:class:`int` or :class:`tuple` of :class:`int` s):
            Stride of filter applications :math:`(s_1, s_2, ..., s_N)`.
            ``stride=s`` is equivalent to ``(s, s, ..., s)``.
        pad (:class:`int` or :class:`tuple` of :class:`int` s):
            Spatial padding width for input arrays
            :math:`(p_1, p_2, ..., p_N)`. ``pad=p`` is equivalent to
            ``(p, p, ..., p)``.
        cover_all (bool): If ``True``, all spatial locations are convoluted
            into some output pixels. It may make the output size larger.
            `cover_all` needs to be ``False`` if you want to use cuDNN.
        dilate (:class:`int` or :class:`tuple` of :class:`int` s):
            Dilation factor of filter applications.
            ``dilate=d`` and ``dilate=(d, d, ..., d)`` are equivalent.
        groups (:class:`int`):
            The number of groups to use grouped convolution.
            The default is one, where grouped convolution is not used.

    Returns:
        ~chainer.Variable:
            Output variable of shape :math:`(n, c_O, l_1, l_2, ..., l_N)`.

    .. note::

        This function uses cuDNN implementation for its forward and backward
        computation if ALL of the following conditions are satisfied:

        - ``cuda.cudnn_enabled`` is ``True``
        - ``chainer.config.use_cudnn`` is ``'always'`` or ``'auto'``
        - The number of spatial dimensions is more than one.
        - ``cover_all`` is ``False``
        - The input's ``dtype`` is equal to the filter weight's.
        - The ``dtype`` is FP16, FP32 or FP64. (FP16 is only available when
          cuDNN version :math:`\\geq` v3.)

    Convolution links can use a feature of cuDNN called autotuning, which
    selects the most efficient CNN algorithm for images of fixed-size,
    can provide a significant performance boost for fixed neural nets.
    To enable, set `chainer.using_config('autotune', True)`

    .. seealso:: :class:`~chainer.links.ConvolutionND`, :func:`convolution_2d`

    .. admonition:: Example

        >>> n = 10
        >>> c_i, c_o = 3, 1
        >>> d1, d2, d3 = 30, 40, 50
        >>> k1, k2, k3 = 10, 10, 10
        >>> p1, p2, p3 = 5, 5, 5
        >>> x = np.random.uniform(0, 1, (n, c_i, d1, d2, d3)).\
astype(np.float32)
        >>> x.shape
        (10, 3, 30, 40, 50)
        >>> W = np.random.uniform(0, 1, (c_o, c_i, k1, k2, k3)).\
astype(np.float32)
        >>> W.shape
        (1, 3, 10, 10, 10)
        >>> b = np.random.uniform(0, 1, (c_o)).astype(np.float32)
        >>> b.shape
        (1,)
        >>> s1, s2, s3 = 2, 4, 6
        >>> y = F.convolution_nd(x, W, b, stride=(s1, s2, s3),\
 pad=(p1, p2, p3))
        >>> y.shape
        (10, 1, 16, 11, 9)
        >>> l1 = int((d1 + 2 * p1 - k1) / s1 + 1)
        >>> l2 = int((d2 + 2 * p2 - k2) / s2 + 1)
        >>> l3 = int((d3 + 2 * p3 - k3) / s3 + 1)
        >>> y.shape == (n, c_o, l1, l2, l3)
        True
        >>> y = F.convolution_nd(x, W, b, stride=(s1, s2, s3),\
 pad=(p1, p2, p3), cover_all=True)
        >>> y.shape == (n, c_o, l1, l2, l3 + 1)
        True

    """
    ndim = len(x.shape[2:])
    fnode = ConvolutionND(
        ndim, stride, pad, cover_all, dilate=dilate, groups=groups)
    args = (x, W) if b is None else (x, W, b)
    y, = fnode.apply(args)
    return y


def convolution_1d(x, W, b=None, stride=1, pad=0, cover_all=False,
                   dilate=1, groups=1):
    """1-dimensional convolution function.

    .. note::

        This function calls :func:`~chainer.functions.convolution_nd`
        internally, so see the details of the behavior in
        the documentation of :func:`~chainer.functions.convolution_nd`.

    """
    if len(x.shape[2:]) != 1:
        raise ValueError(
            'The number of dimensions under channel dimension of the input '
            '\'x\' should be 1. But the actual ndim was {}.'.format(
                len(x.shape[2:])))
    return convolution_nd(x, W, b, stride, pad, cover_all, dilate, groups)


def convolution_3d(x, W, b=None, stride=1, pad=0, cover_all=False,
                   dilate=1, groups=1):
    """3-dimensional convolution function.

    .. note::

        This function calls :func:`~chainer.functions.convolution_nd`
        internally, so see the details of the behavior in
        the documentation of :func:`~chainer.functions.convolution_nd`.

    """
    if len(x.shape[2:]) != 3:
        raise ValueError(
            'The number of dimensions under channel dimension of the input '
            '\'x\' should be 3. But the actual ndim was {}.'.format(
                len(x.shape[2:])))
    return convolution_nd(x, W, b, stride, pad, cover_all, dilate, groups)
