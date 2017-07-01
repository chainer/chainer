import numpy

import chainer
from chainer import configuration
from chainer import cuda
from chainer import function
from chainer.functions.connection import convolution_2d
from chainer.utils import argument
from chainer.utils import conv
from chainer.utils import type_check
from chainer import variable

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _fwd_pref = libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
    if _cudnn_version >= 3000:
        _bwd_filter_pref = \
            libcudnn.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
        _bwd_data_pref = \
            libcudnn.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT


_check_cudnn_acceptable_type = convolution_2d._check_cudnn_acceptable_type


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class Deconvolution2DFunction(function.Function):

    def __init__(self, stride=1, pad=0, outsize=None, requires_x_grad=True,
                 **kwargs):
        argument.check_unexpected_kwargs(
            kwargs, deterministic="deterministic argument is not "
            "supported anymore. "
            "Use chainer.using_config('cudnn_deterministic', value) "
            "context where value is either `True` or `False`.")
        argument.assert_kwargs_empty(kwargs)

        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.outh, self.outw = (None, None) if outsize is None else outsize
        self.requires_x_grad = requires_x_grad

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 4,
            w_type.ndim == 4,
            x_type.shape[1] == w_type.shape[0]
        )

        if self.outh is not None:
            type_check.expect(
                x_type.shape[2] ==
                conv.get_conv_outsize(self.outh, w_type.shape[2],
                                      self.sy, self.ph),
            )
        if self.outw is not None:
            type_check.expect(
                x_type.shape[3] ==
                conv.get_conv_outsize(self.outw, w_type.shape[3],
                                      self.sx, self.pw),
            )

        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[1]
            )

    def forward_cpu(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None

        if not type_check.same_types(*inputs):
            if b is not None:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}, type(b): {2}'
                                 .format(type(W), type(x), type(b)))
            else:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}'
                                 .format(type(W), type(x)))

        kh, kw = W.shape[2:]
        _, _, h, w = x.shape
        gcol = numpy.tensordot(W, x, (0, 1)).astype(x.dtype, copy=False)
        # - k, m, n: shape of out_channel
        # - b: number of inputs
        # - h, w: height and width of kernels
        # k, m, n, b, h, w -> b, k, m, n, h, w
        gcol = numpy.rollaxis(gcol, 3)
        if self.outh is None:
            self.outh = conv.get_deconv_outsize(h, kh, self.sy, self.ph)
            assert self.outh > 0, 'Height in the output should be positive.'
        if self.outw is None:
            self.outw = conv.get_deconv_outsize(w, kw, self.sx, self.pw)
            assert self.outw > 0, 'Width in the output should be positive.'
        y = conv.col2im_cpu(
            gcol, self.sy, self.sx, self.ph, self.pw, self.outh, self.outw)
        # b, k, h, w
        if b is not None:
            y += b.reshape(1, b.size, 1, 1)
        return y,

    def forward_gpu(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None

        if not type_check.same_types(*inputs):
            if b is not None:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}, type(b): {2}'
                                 .format(type(W), type(x), type(b)))
            else:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}'
                                 .format(type(W), type(x)))

        kh, kw = W.shape[2:]
        n, in_c, in_h, in_w = x.shape
        c = W.shape[1]  # out_c
        if self.outh is None:
            self.outh = conv.get_deconv_outsize(in_h, kh, self.sy, self.ph)
            assert self.outh > 0, 'Height in the output should be positive.'
        if self.outw is None:
            self.outw = conv.get_deconv_outsize(in_w, kw, self.sx, self.pw)
            assert self.outw > 0, 'Width in the output should be positive.'
        if (chainer.should_use_cudnn('>=auto') and
                _check_cudnn_acceptable_type(x.dtype, W.dtype)):
            x = cuda.cupy.ascontiguousarray(x)
            W = cuda.cupy.ascontiguousarray(W)
            if b is not None:
                b = cuda.cupy.ascontiguousarray(b)

            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x)
            y = cuda.cupy.empty((n, c, self.outh, self.outw),
                                dtype=x.dtype)
            y_desc = cudnn.create_tensor_descriptor(y)

            self.filter_desc = cudnn.create_filter_descriptor(W)
            self.conv_desc = cudnn.create_convolution_descriptor(
                (self.ph, self.pw), (self.sy, self.sx), x.dtype)
            if b is not None:
                self.bias_desc = cudnn.create_tensor_descriptor(
                    b[None, :, None, None])

            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes

            if _cudnn_version >= 3000:
                workspace_size = cuda.get_max_workspace_size()
                workspace = cuda.cupy.empty((workspace_size,), dtype='b')
                if configuration.config.cudnn_deterministic:
                    algo = cuda.cupy.cuda.cudnn.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1  # NOQA
                else:
                    algo = libcudnn.getConvolutionBackwardDataAlgorithm(
                        handle, self.filter_desc.value, x_desc.value,
                        self.conv_desc.value, y_desc.value, _bwd_data_pref,
                        workspace_size)

                libcudnn.convolutionBackwardData_v3(
                    handle, one.data, self.filter_desc.value, W.data.ptr,
                    x_desc.value, x.data.ptr, self.conv_desc.value,
                    algo, workspace.data.ptr, workspace_size,
                    zero.data, y_desc.value, y.data.ptr)
            else:
                if configuration.config.cudnn_deterministic:
                    raise ValueError(
                        "`cudnn_deterministic` option must be False "
                        "if the forward propagation of "
                        "chainer.functions.Deconvolution2D "
                        "uses cuDNN and cuDNN versions < v3. "
                        "Turn off cudnn_deterministic option with "
                        "`chainer.using_config('cudnn_deterministic', False)` "
                        "context.")
                libcudnn.convolutionBackwardData_v2(
                    handle, one.data, self.filter_desc.value, W.data.ptr,
                    x_desc.value, x.data.ptr, self.conv_desc.value,
                    zero.data, y_desc.value, y.data.ptr)

            if b is not None:
                cudnn.add_tensor(
                    handle, one.data, self.bias_desc.value, b.data.ptr,
                    one.data, y_desc.value, y.data.ptr)
        else:
            gcol = cuda.cupy.tensordot(W, x, (0, 1)).astype(x.dtype,
                                                            copy=False)
            # - k, m, n: shape of out_channel
            # - b: number of inputs
            # - h, w: height and width of kernels
            # k, m, n, b, h, w -> b, k, m, n, h, w
            gcol = cuda.cupy.rollaxis(gcol, 3)
            y = conv.col2im_gpu(
                gcol, self.sy, self.sx, self.ph, self.pw, self.outh, self.outw)
            if b is not None:
                y += b.reshape(1, b.size, 1, 1)
        return y,

    def backward_cpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None

        if not type_check.same_types(*inputs):
            if b is not None:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}, type(b): {2}'
                                 .format(type(W), type(x), type(b)))
            else:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}'
                                 .format(type(W), type(x)))

        gy = grad_outputs[0]
        kh, kw = W.shape[2:]
        col = conv.im2col_cpu(
            gy, kh, kw, self.sy, self.sx, self.ph, self.pw)
        gW = numpy.tensordot(
            x, col, ([0, 2, 3], [0, 4, 5])).astype(W.dtype, copy=False)
        if not self.requires_x_grad:
            gx = None
        else:
            gx = numpy.tensordot(
                col, W, ([1, 2, 3], [1, 2, 3])).astype(x.dtype, copy=False)
            gx = numpy.rollaxis(gx, 3, 1)

        if b is None:
            return gx, gW
        else:
            gb = gy.sum(axis=(0, 2, 3))
            return gx, gW, gb

    def backward_gpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None

        if not type_check.same_types(*inputs):
            if b is not None:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}, type(b): {2}'
                                 .format(type(W), type(x), type(b)))
            else:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}'
                                 .format(type(W), type(x)))

        gy = grad_outputs[0]
        n, in_c, in_h, in_w = x.shape
        _, out_channels, kh, kw = W.shape
        c, h, w = gy.shape[1:]
        gx = None

        if (chainer.should_use_cudnn('>=auto') and
                _check_cudnn_acceptable_type(x.dtype, W.dtype)):
            gx = cuda.cupy.empty((n, in_c, in_h, in_w), dtype=x.dtype)
            x = cuda.cupy.ascontiguousarray(x)
            W = cuda.cupy.ascontiguousarray(W)
            gy = cuda.cupy.ascontiguousarray(gy)
            if b is not None:
                b = cuda.cupy.ascontiguousarray(b)

            handle = cudnn.get_handle()
            gy_desc = cudnn.create_tensor_descriptor(gy)
            gx_desc = cudnn.create_tensor_descriptor(gx)

            # chance to choose implicit-precomp-gemm algorithm
            workspace_size = cuda.get_max_workspace_size()
            algo = libcudnn.getConvolutionForwardAlgorithm(
                handle, gy_desc.value, self.filter_desc.value,
                self.conv_desc.value, gx_desc.value, _fwd_pref,
                workspace_size)
            workspace = cuda.cupy.empty((workspace_size,), dtype='b')

            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes

            libcudnn.convolutionForward(
                handle, one.data, gy_desc.value, gy.data.ptr,
                self.filter_desc.value, W.data.ptr,
                self.conv_desc.value, algo, workspace.data.ptr, workspace_size,
                zero.data, gx_desc.value, gx.data.ptr)
            # bias backward
            if b is not None:
                gb = cuda.cupy.empty_like(b)
                libcudnn.convolutionBackwardBias(
                    handle, one.data, gy_desc.value, gy.data.ptr,
                    zero.data, self.bias_desc.value, gb.data.ptr)
            gW = cuda.cupy.empty_like(W)
            # filter backward
            if _cudnn_version >= 3000:
                if configuration.config.cudnn_deterministic:
                    algo = cuda.cupy.cuda.cudnn.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1  # NOQA
                else:
                    algo = libcudnn.getConvolutionBackwardFilterAlgorithm(
                        handle, gy_desc.value, gx_desc.value,
                        self.conv_desc.value, self.filter_desc.value,
                        _bwd_filter_pref, workspace_size)

                libcudnn.convolutionBackwardFilter_v3(
                    handle, one.data, gy_desc.value, gy.data.ptr,
                    gx_desc.value, x.data.ptr, self.conv_desc.value,
                    algo, workspace.data.ptr, workspace_size,
                    zero.data, self.filter_desc.value, gW.data.ptr)
            else:
                if configuration.config.cudnn_deterministic:
                    raise ValueError(
                        "`cudnn_deterministic` option must be False "
                        "if the backpropagation of "
                        "chainer.functions.Deconvolution2D "
                        "uses cuDNN and cuDNN versions < v3. "
                        "Turn off cudnn_deterministic option with "
                        "`chainer.using_config('cudnn_deterministic', False)` "
                        "context.")
                libcudnn.convolutionBackwardFilter_v2(
                    handle, one.data, gy_desc.value, gy.data.ptr,
                    gx_desc.value, x.data.ptr, self.conv_desc.value,
                    zero.data, self.filter_desc.value, gW.data.ptr)
        else:
            # Implementation using im2col
            col = conv.im2col_gpu(
                gy, kh, kw, self.sy, self.sx, self.ph, self.pw)

            gW = cuda.cupy.tensordot(
                x, col, ([0, 2, 3], [0, 4, 5])).astype(W.dtype, copy=False)
            if self.requires_x_grad:
                gx = cuda.cupy.tensordot(
                    col, W, ([1, 2, 3], [1, 2, 3])).astype(x.dtype, copy=False)
                gx = cuda.cupy.rollaxis(gx, 3, 1)

            # bias backward
            if b is not None:
                gb = gy.sum(axis=(0, 2, 3))

        if b is None:
            return gx, gW
        else:
            return gx, gW, gb


def deconvolution_2d(x, W, b=None, stride=1, pad=0, outsize=None, **kwargs):
    """deconvolution_2d(x, W, b=None, stride=1, pad=0, outsize=None)

    Two dimensional deconvolution function.

    This is an implementation of two-dimensional deconvolution. In most of deep
    learning frameworks and papers, this function is called
    **transposed convolution**. But because of historical reasons (e.g. paper
    by Ziller `Deconvolutional Networks`_) and backward compatibility, this
    function is called **deconvolution** in Chainer.

    .. _Deconvolutional Networks: \
http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf

    It takes three variables: input image ``x``,
    the filter weight ``W``, and the bias vector ``b``.

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

    Let :math:`(s_Y, s_X)` be the stride of filter application. Then, the
    output size :math:`(h_O, w_O)` is estimated by the following equations:

    .. math::

       h_O &= s_Y (h_I - 1) + h_K - 2h_P,\\\\
       w_O &= s_X (w_I - 1) + w_K - 2w_P.

    The output of this function can be non-deterministic when it uses cuDNN.
    If ``chainer.configuration.config.deterministic`` is ``True`` and
    cuDNN version is >= v3, it forces cuDNN to use a deterministic algorithm.

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
            Weight variable of shape :math:`(c_I, c_O, h_K, w_K)`.
        b (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Bias variable of length :math:`c_O` (optional).
        stride (:class:`int` or pair of :class:`int` s):
            Stride of filter applications. ``stride=s`` and ``stride=(s, s)``
            are equivalent.
        pad (:class:`int` or pair of :class:`int` s):
            Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        outsize (:class:`tuple` of :class:`int`):
            Expected output size of deconvolutional operation.
            It should be pair of height and width :math:`(h_O, w_O)`.
            Default value is ``None`` and the outsize is estimated by
            input size, stride and pad.

    Returns:
        ~chainer.Variable:
            Output variable of shape :math:`(n, c_O, h_O, w_O)`.

    .. admonition:: Example

        >>> n = 10
        >>> c_i, c_o = 1, 3
        >>> h_i, w_i = 5, 10
        >>> h_k, w_k = 10, 10
        >>> h_p, w_p = 5, 5
        >>> x = np.random.uniform(0, 1, (n, c_i, h_i, w_i)).astype('f')
        >>> x.shape
        (10, 1, 5, 10)
        >>> W = np.random.uniform(0, 1, (c_i, c_o, h_k, w_k)).astype('f')
        >>> W.shape
        (1, 3, 10, 10)
        >>> b = np.random.uniform(0, 1, c_o).astype('f')
        >>> b.shape
        (3,)
        >>> s_y, s_x = 5, 5
        >>> y = F.deconvolution_2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
        >>> y.shape
        (10, 3, 20, 45)
        >>> h_o = s_y * (h_i - 1) + h_k - 2 * h_p
        >>> w_o = s_x * (w_i - 1) + w_k - 2 * w_p
        >>> y.shape == (n, c_o, h_o, w_o)
        True


    """
    argument.check_unexpected_kwargs(
        kwargs, deterministic="deterministic argument is not "
        "supported anymore. "
        "Use chainer.using_config('cudnn_deterministic', value) "
        "context where value is either `True` or `False`.")
    argument.assert_kwargs_empty(kwargs)

    requires_x_grad = isinstance(x, variable.Variable) and x.requires_grad
    func = Deconvolution2DFunction(stride, pad, outsize, requires_x_grad)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)
