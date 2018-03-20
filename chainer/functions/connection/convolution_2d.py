import numpy
import six

import chainer
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import configuration
from chainer import function_node
import chainer.functions
from chainer.utils import argument
from chainer.utils import conv
from chainer.utils import type_check

if cuda.cudnn_enabled:
    _cudnn_version = cuda.cuda.cudnn.getVersion()


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class Convolution2DFunction(function_node.FunctionNode):

    _use_ideep = False

    def __init__(self, stride=1, pad=0, cover_all=False, groups=1, **kwargs):
        argument.check_unexpected_kwargs(
            kwargs,
            deterministic="deterministic argument is not supported anymore. "
            "Use chainer.using_config('cudnn_deterministic', value) context "
            "where value is either `True` or `False`.",
            requires_x_grad="requires_x_grad argument is not supported "
            "anymore. Just remove the argument. Note that whether to compute "
            "the gradient w.r.t. x is automatically decided during "
            "backpropagation."
        )
        dilate, = argument.parse_kwargs(kwargs, ('dilate', 1))

        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.cover_all = cover_all
        self.dy, self.dx = _pair(dilate)
        self.groups = groups

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 4,
            w_type.ndim == 4,
            # Need to consider the case that group count > 1.
            # x_type.shape[1] == w_type.shape[1],
        )

        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def _get_out_size(self, inputs):
        x, W = inputs[:2]
        _, _, kh, kw = W.shape
        _, _, h, w = x.shape
        out_h = conv.get_conv_outsize(
            h, kh, self.sy, self.ph, cover_all=self.cover_all, d=self.dy)
        if out_h <= 0:
            raise RuntimeError('Height in the output should be positive.')
        out_w = conv.get_conv_outsize(
            w, kw, self.sx, self.pw, cover_all=self.cover_all, d=self.dx)
        if out_w <= 0:
            raise RuntimeError('Width in the output should be positive.')
        return out_h, out_w

    def forward_cpu(self, inputs):
        if (self.groups == 1
                and intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(inputs)):
            # iDeep implementation
            self._use_ideep = True
            return self._forward_ideep(inputs)

        self.retain_inputs((0, 1))  # retain only x and W
        if len(inputs) == 2:
            (x, W), b = inputs, None
        else:
            x, W, b = inputs

        if self.groups > 1:
            return self._forward_grouped_convolution(x, W, b)
        else:
            return self._forward_cpu_core(x, W, b)

    def _forward_cpu_core(self, x, W, b):
        kh, kw = W.shape[2:]
        col = conv.im2col_cpu(
            x, kh, kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        y = numpy.tensordot(
            col, W, ((1, 2, 3), (1, 2, 3))).astype(x.dtype, copy=False)
        if b is not None:
            y += b
        y = numpy.rollaxis(y, 3, 1)
        return y,

    def _forward_ideep(self, inputs):
        self.retain_inputs((0, 1))
        if len(inputs) == 3:
            x, W, b = inputs
        else:
            (x, W), b = inputs, None
        out_c, input_c, kh, kw = W.shape
        n, c, h, w = x.shape

        out_h, out_w = self._get_out_size(inputs)
        pd = (self.sy * (out_h - 1)
              + (kh + (kh - 1) * (self.dy - 1)) - h - self.ph)
        pr = (self.sx * (out_w - 1)
              + (kw + (kw - 1) * (self.dx - 1)) - w - self.pw)
        param = intel64.ideep.convolution2DParam(
            (n, out_c, out_h, out_w),
            self.dy, self.dx,
            self.sy, self.sx,
            self.ph, self.pw,
            pd, pr)
        y = intel64.ideep.convolution2D.Forward(
            intel64.ideep.array(x),
            intel64.ideep.array(W),
            intel64.ideep.array(b) if b is not None else None,
            param)
        return y,

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))  # retain only x and W
        if len(inputs) == 2:
            (x, W), b = inputs, None
        else:
            x, W, b = inputs

        out_c, _, kh, kw = W.shape
        n, _, h, w = x.shape

        out_h, out_w = self._get_out_size(inputs)
        y = cuda.cupy.empty((n, out_c, out_h, out_w), dtype=x.dtype)

        use_cudnn = (
            chainer.should_use_cudnn('>=auto')
            and not self.cover_all
            and x.dtype == W.dtype
            and ((self.dy == 1 and self.dx == 1) or _cudnn_version >= 6000)
            and (self.groups <= 1 or _cudnn_version >= 7000)
        )

        if use_cudnn:
            # cuDNN implementation
            return self._forward_cudnn(x, W, b, y)

        elif self.groups > 1:
            return self._forward_grouped_convolution(x, W, b)

        else:
            return self._forward_gpu_core(x, W, b)

    def _forward_gpu_core(self, x, W, b):
        kh, kw = W.shape[2:]
        # Implementation using im2col
        col = conv.im2col_gpu(
            x, kh, kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        y = cuda.cupy.tensordot(
            col, W, ((1, 2, 3), (1, 2, 3))).astype(x.dtype, copy=False)
        # TODO(beam2d): Support unshared bias
        if b is not None:
            y += b
        y = cuda.cupy.rollaxis(y, 3, 1)
        return y,

    def _forward_grouped_convolution(self, x, W, b):
        # G: group count
        # N: batch size
        # kH, kW: kernel height, kernel width
        # iC, iH, iW: input channels, input height, input width
        # oC, oH, oW: output channels, output height, output width
        G = self.groups
        N, iC, iH, iW = x.shape
        oC, _, kH, kW = W.shape
        iCg = int(iC / G)
        oCg = int(oC / G)

        xp = cuda.get_array_module(x)

        _x = x.reshape(N, G, iCg, iH, iW)
        _x = xp.rollaxis(_x, 1)  # (G, N, iCg, iH, iW)
        _W = W.reshape(G, oCg, iCg, kH, kW)
        if b is not None:
            _b = b.reshape(G, oCg)

        _ys = []
        for g in six.moves.range(G):
            _bg = None if b is None else _b[g, ]
            if xp is numpy:
                _y, = self._forward_cpu_core(_x[g, ], _W[g, ], _bg)
            else:
                _y, = self._forward_gpu_core(_x[g, ], _W[g, ], _bg)
            _ys.append(_y)

        y = xp.concatenate(_ys, axis=1)  # (N, oC, oH, oW)
        return y,

    def _forward_cudnn(self, x, W, b, y):
        pad = (self.ph, self.pw)
        stride = (self.sy, self.sx)
        dilation = (self.dy, self.dx)
        auto_tune = configuration.config.autotune
        tensor_core = configuration.config.use_cudnn_tensor_core
        cuda.cudnn.convolution_forward(
            x, W, b, y, pad, stride, dilation, self.groups,
            auto_tune=auto_tune, tensor_core=tensor_core)
        return y,

    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        gy, = grad_outputs

        ret = []
        if 0 in indexes:
            xh, xw = x.shape[2:]
            gx = chainer.functions.deconvolution_2d(
                gy, W, stride=(self.sy, self.sx), pad=(self.ph, self.pw),
                outsize=(xh, xw), dilate=(self.dy, self.dx),
                groups=self.groups)
            ret.append(gx)
        if 1 in indexes:
            gW, = Convolution2DGradW(self).apply((x, gy))
            ret.append(gW)
        if 2 in indexes:
            gb = chainer.functions.sum(gy, axis=(0, 2, 3))
            ret.append(gb)

        return ret


class Convolution2DGradW(function_node.FunctionNode):

    def __init__(self, conv2d):
        W_node = conv2d.inputs[1]
        self.kh, self.kw = W_node.shape[2:]
        self.sy = conv2d.sy
        self.sx = conv2d.sx
        self.ph = conv2d.ph
        self.pw = conv2d.pw
        self.dy = conv2d.dy
        self.dx = conv2d.dx
        self.cover_all = conv2d.cover_all
        self.W_dtype = W_node.dtype
        self.groups = conv2d.groups
        self._use_ideep = conv2d._use_ideep
        assert self.groups == 1 or not self._use_ideep

    def forward_cpu(self, inputs):
        if self._use_ideep:
            return self._forward_ideep(inputs)

        self.retain_inputs((0, 1))
        x, gy = inputs

        # NumPy raises an error when the array is not contiguous.
        # See: https://github.com/chainer/chainer/issues/2744
        # TODO(niboshi): Remove this code when NumPy is fixed.
        if (not (gy.flags.c_contiguous or gy.flags.f_contiguous) and
                1 in gy.shape):
            gy = numpy.ascontiguousarray(gy)

        if self.groups > 1:
            return self._forward_grouped_convolution(x, gy)
        else:
            return self._forward_cpu_core(x, gy)

    def _forward_cpu_core(self, x, gy):
        col = conv.im2col_cpu(
            x, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        gW = numpy.tensordot(gy, col, ((0, 2, 3), (0, 4, 5))
                             ).astype(self.W_dtype, copy=False)
        return gW,

    def _forward_ideep(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs

        n, input_c, h, w = x.shape
        n, out_c, out_h, out_w = gy.shape
        pd = (self.sy * (out_h - 1)
              + (self.kh + (self.kh - 1) * (self.dy - 1))
              - h - self.ph)
        pr = (self.sx * (out_w - 1)
              + (self.kw + (self.kw - 1) * (self.dx - 1))
              - w - self.pw)

        param = intel64.ideep.convolution2DParam(
            (out_c, input_c, self.kh, self.kw),
            self.dy, self.dx,
            self.sy, self.sx,
            self.ph, self.pw,
            pd, pr)
        gW = intel64.ideep.convolution2D.BackwardWeights(
            intel64.ideep.array(x),
            intel64.ideep.array(gy),
            param)
        return gW,

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs

        use_cudnn = (
            chainer.should_use_cudnn('>=auto')
            and not self.cover_all
            and x.dtype == self.W_dtype
            and ((self.dy == 1 and self.dx == 1)
                 or (_cudnn_version >= 6000
                     and not configuration.config.cudnn_deterministic))
            and (self.groups <= 1 or _cudnn_version >= 7000)
        )

        if use_cudnn:
            # cuDNN implementation
            return self._forward_cudnn(x, gy)

        elif self.groups > 1:
            return self._forward_grouped_convolution(x, gy)

        else:
            return self._forward_gpu_core(x, gy)

    def _forward_gpu_core(self, x, gy):
        col = conv.im2col_gpu(
            x, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        gW = cuda.cupy.tensordot(gy, col, ((0, 2, 3), (0, 4, 5))
                                 ).astype(self.W_dtype, copy=False)
        return gW,

    def _forward_grouped_convolution(self, x, gy):
        # G: group count
        # N: batch size
        # kH, kW: kernel height, kernel width
        # iC, iH, iW: input channels, input height, input width
        # oC, oH, oW: output channels, output height, output width
        G = self.groups
        N, iC, iH, iW = x.shape
        _, oC, oH, oW = gy.shape
        iCg = int(iC / G)
        oCg = int(oC / G)

        xp = cuda.get_array_module(x)

        _x = x.reshape(N, G, iCg, iH, iW)
        _x = xp.rollaxis(_x, 1)  # (G, N, iCg, iH, iW)
        _gy = gy.reshape(N, G, oCg, oH, oW)
        _gy = xp.rollaxis(_gy, 1)  # (G, N, oCg, oH, oW)
        # Work-around for NumPy's bug?
        if xp is numpy:
            _gy = xp.ascontiguousarray(_gy)

        _gWs = []
        for g in six.moves.range(G):
            if xp is numpy:
                _gW, = self._forward_cpu_core(_x[g, ], _gy[g, ])
            else:
                _gW, = self._forward_gpu_core(_x[g, ], _gy[g, ])
            _gWs.append(_gW)

        gW = xp.concatenate(_gWs)  # (oC, iCg, kH, kW)
        return gW,

    def _forward_cudnn(self, x, gy):
        _, out_c, out_h, out_w = gy.shape
        n, c, h, w = x.shape

        iC = c
        iCg = int(iC / self.groups)
        gW = cuda.cupy.empty((out_c, iCg, self.kh, self.kw),
                             dtype=self.W_dtype)
        pad = (self.ph, self.pw)
        stride = (self.sy, self.sx)
        dilation = (self.dy, self.dx)
        deterministic = configuration.config.cudnn_deterministic
        auto_tune = configuration.config.autotune
        tensor_core = configuration.config.use_cudnn_tensor_core
        cuda.cudnn.convolution_backward_filter(
            x, gy, gW, pad, stride, dilation, self.groups,
            deterministic=deterministic, auto_tune=auto_tune,
            tensor_core=tensor_core)

        return gW,

    def backward(self, indexes, grad_outputs):
        x, gy = self.get_retained_inputs()
        ggW, = grad_outputs

        ret = []
        if 0 in indexes:
            xh, xw = x.shape[2:]
            gx = chainer.functions.deconvolution_2d(
                gy, ggW, stride=(self.sy, self.sx), pad=(self.ph, self.pw),
                outsize=(xh, xw), dilate=(self.dy, self.dx),
                groups=self.groups)
            ret.append(gx)
        if 1 in indexes:
            ggy = convolution_2d(
                x, ggW, stride=(self.sy, self.sx), pad=(self.ph, self.pw),
                cover_all=self.cover_all, dilate=(self.dy, self.dx),
                groups=self.groups)
            ret.append(ggy)

        return ret


def convolution_2d(x, W, b=None, stride=1, pad=0, cover_all=False, groups=1,
                   **kwargs):
    """convolution_2d(x, W, b=None, stride=1, pad=0, cover_all=False, *, dilate=1)

    Two-dimensional convolution function.

    This is an implementation of two-dimensional convolution in ConvNets.
    It takes three variables: the input image ``x``, the filter weight ``W``,
    and the bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` and :math:`c_O` are the number of the input and output
      channels, respectively.
    - :math:`h_I` and :math:`w_I` are the height and width of the input image,
      respectively.
    - :math:`h_K` and :math:`w_K` are the height and width of the filters,
      respectively.
    - :math:`h_P` and :math:`w_P` are the height and width of the spatial
      padding size, respectively.

    Then the ``Convolution2D`` function computes correlations between filters
    and patches of size :math:`(h_K, w_K)` in ``x``.
    Note that correlation here is equivalent to the inner product between
    expanded vectors.
    Patches are extracted at positions shifted by multiples of ``stride`` from
    the first position ``(-h_P, -w_P)`` for each spatial axis.
    The right-most (or bottom-most) patches do not run over the padded spatial
    size.

    Let :math:`(s_Y, s_X)` be the stride of filter application. Then, the
    output size :math:`(h_O, w_O)` is determined by the following equations:

    .. math::

       h_O &= (h_I + 2h_P - h_K) / s_Y + 1,\\\\
       w_O &= (w_I + 2w_P - w_K) / s_X + 1.

    If ``cover_all`` option is ``True``, the filter will cover the all
    spatial locations. So, if the last stride of filter does not cover the
    end of spatial locations, an addtional stride will be applied to the end
    part of spatial locations. In this case, the output size :math:`(h_O, w_O)`
    is determined by the following equations:

    .. math::

       h_O &= (h_I + 2h_P - h_K + s_Y - 1) / s_Y + 1,\\\\
       w_O &= (w_I + 2w_P - w_K + s_X - 1) / s_X + 1.

    If the bias vector is given, then it is added to all spatial locations of
    the output of convolution.

    The output of this function can be non-deterministic when it uses cuDNN.
    If ``chainer.configuration.config.cudnn_deterministic`` is ``True`` and
    cuDNN version is >= v3, it forces cuDNN to use a deterministic algorithm.

    Convolution links can use a feature of cuDNN called autotuning, which
    selects the most efficient CNN algorithm for images of fixed-size,
    can provide a significant performance boost for fixed neural nets.
    To enable, set `chainer.using_config('autotune', True)`

    When the dilation factor is greater than one, cuDNN is not used unless
    the version is 6.0 or higher.

    .. warning::

        ``deterministic`` argument is not supported anymore since v2.
        Instead, use ``chainer.using_config('cudnn_deterministic', value)``
        (value is either ``True`` or ``False``).
        See :func:`chainer.using_config`.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable of shape :math:`(n, c_I, h_I, w_I)`.
        W (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Weight variable of shape :math:`(c_O, c_I, h_K, w_K)`.
        b (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Bias variable of length :math:`c_O` (optional).
        stride (:class:`int` or pair of :class:`int` s):
            Stride of filter applications. ``stride=s`` and ``stride=(s, s)``
            are equivalent.
        pad (:class:`int` or pair of :class:`int` s):
            Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        cover_all (bool): If ``True``, all spatial locations are convoluted
            into some output pixels.
        dilate (int or pair of ints): Dilation factor of filter applications.
            ``dilate=d`` and ``dilate=(d, d)`` are equivalent.

    Returns:
        ~chainer.Variable:
            Output variable of shape :math:`(n, c_O, h_O, w_O)`.

    .. seealso:: :class:`~chainer.links.Convolution2D`

    .. admonition:: Example

        >>> n = 10
        >>> c_i, c_o = 3, 1
        >>> h_i, w_i = 30, 40
        >>> h_k, w_k = 10, 10
        >>> h_p, w_p = 5, 5
        >>> x = np.random.uniform(0, 1, (n, c_i, h_i, w_i)).astype(np.float32)
        >>> x.shape
        (10, 3, 30, 40)
        >>> W = np.random.uniform(0, 1, (c_o, c_i, h_k, w_k)).\
astype(np.float32)
        >>> W.shape
        (1, 3, 10, 10)
        >>> b = np.random.uniform(0, 1, (c_o,)).astype(np.float32)
        >>> b.shape
        (1,)
        >>> s_y, s_x = 5, 7
        >>> y = F.convolution_2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
        >>> y.shape
        (10, 1, 7, 6)
        >>> h_o = int((h_i + 2 * h_p - h_k) / s_y + 1)
        >>> w_o = int((w_i + 2 * w_p - w_k) / s_x + 1)
        >>> y.shape == (n, c_o, h_o, w_o)
        True
        >>> y = F.convolution_2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p), \
cover_all=True)
        >>> y.shape == (n, c_o, h_o, w_o + 1)
        True

    """
    argument.check_unexpected_kwargs(
        kwargs, deterministic="deterministic argument is not "
        "supported anymore. "
        "Use chainer.using_config('cudnn_deterministic', value) "
        "context where value is either `True` or `False`.")
    dilate, = argument.parse_kwargs(kwargs, ('dilate', 1))

    fnode = Convolution2DFunction(stride, pad, cover_all, dilate=dilate,
                                  groups=groups)
    if b is None:
        args = x, W
    else:
        args = x, W, b
    y, = fnode.apply(args)
    return y
