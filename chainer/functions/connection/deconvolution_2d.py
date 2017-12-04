import numpy

import chainer
from chainer.backends import cuda
from chainer import configuration
from chainer import function_node
import chainer.functions
from chainer.functions.connection import convolution_2d
from chainer.utils import argument
from chainer.utils import conv
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cuda.cudnn
    _cudnn_version_ = libcudnn.getVersion()
    _fwd_pref = libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
    _bwd_filter_pref = \
        libcudnn.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
    _bwd_data_pref = \
        libcudnn.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT
    _algorithm = {}


def get_algorithm(W, dy, dx, conv_param, handle, filter_desc, dy_desc,
                  conv_desc, dx_desc, workspace):
    key = (dx.shape, W.shape, dy.shape, conv_param)
    if key in _algorithm:
        return _algorithm[key]
    ret = libcudnn.findConvolutionBackwardDataAlgorithmEx(
        handle, filter_desc.value, W.data.ptr, dy_desc.value, dy.data.ptr,
        conv_desc.value, dx_desc.value, dx.data.ptr, 1, workspace.data.ptr,
        workspace.size)
    algo = ret[1][0]['algo']
    _algorithm[key] = algo
    return algo


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class Deconvolution2DFunction(function_node.FunctionNode):

    cover_all = None

    def __init__(self, stride=1, pad=0, outsize=None, **kwargs):
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
        argument.assert_kwargs_empty(kwargs)

        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.outh, self.outw = (None, None) if outsize is None else outsize

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
            lower_bound = conv.get_conv_outsize(
                self.outh, w_type.shape[2], self.sy, self.ph)
            upper_bound = conv.get_conv_outsize(
                self.outh, w_type.shape[2], self.sy, self.ph, cover_all=True)
            type_check.expect(
                lower_bound <= x_type.shape[2],
                x_type.shape[2] <= upper_bound)
        if self.outw is not None:
            lower_bound = conv.get_conv_outsize(
                self.outw, w_type.shape[3], self.sx, self.pw)
            upper_bound = conv.get_conv_outsize(
                self.outw, w_type.shape[3], self.sx, self.pw, cover_all=True)
            type_check.expect(
                lower_bound <= x_type.shape[3],
                x_type.shape[3] <= upper_bound)

        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[1]
            )

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))  # only retain x and W
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None

        if not all([isinstance(i, numpy.ndarray) for i in inputs]):
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
        self.retain_inputs((0, 1))  # only retain x and W
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None

        if not all([isinstance(i, cuda.ndarray) for i in inputs]):
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

        self._set_cover_all(x, W)

        if (not self.cover_all and chainer.should_use_cudnn('>=auto') and
                x.dtype == W.dtype):
            x = cuda.cupy.ascontiguousarray(x)
            W = cuda.cupy.ascontiguousarray(W)
            if b is not None:
                b = cuda.cupy.ascontiguousarray(b)

            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x)
            y = cuda.cupy.empty((n, c, self.outh, self.outw),
                                dtype=x.dtype)
            y_desc = cudnn.create_tensor_descriptor(y)

            filter_desc = cudnn.create_filter_descriptor(W)
            conv_param = (self.ph, self.pw), (self.sy, self.sx), x.dtype
            conv_desc = cudnn.create_convolution_descriptor(
                *conv_param)
            if b is not None:
                bias_desc = cudnn.create_tensor_descriptor(
                    b[None, :, None, None])

            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes

            workspace_size = cuda.get_max_workspace_size()
            workspace = cuda.cupy.empty((workspace_size,), dtype='b')

            if configuration.config.cudnn_deterministic:
                algo = libcudnn.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
            elif configuration.config.autotune and _cudnn_version_ >= 5000:
                algo = get_algorithm(
                    W, x, y, conv_param, handle, filter_desc,
                    x_desc, conv_desc, y_desc, workspace)
            else:
                algo = libcudnn.getConvolutionBackwardDataAlgorithm(
                    handle, filter_desc.value, x_desc.value, conv_desc.value,
                    y_desc.value, _bwd_data_pref, workspace_size)

            libcudnn.convolutionBackwardData_v3(
                handle, one.data, filter_desc.value, W.data.ptr,
                x_desc.value, x.data.ptr, conv_desc.value,
                algo, workspace.data.ptr, workspace_size,
                zero.data, y_desc.value, y.data.ptr)

            if b is not None:
                cudnn.add_tensor(
                    handle, one.data, bias_desc.value, b.data.ptr,
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

    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        gy, = grad_outputs

        ret = []
        if 0 in indexes:
            if self.cover_all is None:
                self._set_cover_all(x, W)
            gx = chainer.functions.convolution_2d(
                gy, W, stride=(self.sy, self.sx), pad=(self.ph, self.pw),
                cover_all=self.cover_all)
            ret.append(gx)
        if 1 in indexes:
            if self.cover_all is None:
                self._set_cover_all(x, W)
            gW, = convolution_2d.Convolution2DGradW(self).apply((gy, x))
            ret.append(gW)
        if 2 in indexes:
            gb = chainer.functions.sum(gy, axis=(0, 2, 3))
            ret.append(gb)

        return ret

    def _set_cover_all(self, x, W):
        in_h, in_w = x.shape[2:]
        kh, kw = W.shape[2:]
        self.cover_all = (
            in_h != conv.get_conv_outsize(self.outh, kh, self.sy, self.ph) or
            in_w != conv.get_conv_outsize(self.outw, kw, self.sx, self.pw))


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

    Deconvolution links can use a feature of cuDNN called autotuning, which
    selects the most efficient CNN algorithm for images of fixed-size,
    can provide a significant performance boost for fixed neural nets.
    To enable, set `chainer.using_config('autotune', True)`

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

    func = Deconvolution2DFunction(stride, pad, outsize)
    if b is None:
        args = x, W
    else:
        args = x, W, b
    y, = func.apply(args)
    return y
