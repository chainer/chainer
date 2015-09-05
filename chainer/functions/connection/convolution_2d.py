import ctypes
import math

import numpy
from six import moves

from chainer import cuda
from chainer import function
from chainer import model
from chainer.utils import conv
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn
    _fwd_pref = libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return (x, x)


class Convolution2D(model.Model, function.Function):

    """Two-dimensional convolution function.

    The details of this function are described below the arguments description.

    Args:
        in_channels (int): Number of channels of input arrays.
        out_channels (int): Number of channels of output arrays.
        ksize (int or (int, int)): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or (int, int)): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or (int, int)): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        wscale (float): Scaling factor of the initial weight.
        bias (float): Initial bias value.
        nobias (bool): If True, then this function does not use the bias term.
        use_cudnn (bool): If True, then this function uses CuDNN if available.
        initialW (4-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.
        dtype (numpy.dtype): Type to use in computing.

    This function holds at most two parameter arrays: ``W`` and ``b``, which
    indicate the filter weight and the bias vector, respectively.

    The filter weight has four dimensions :math:`(c_O, c_I, k_H, k_W)`
    which indicate the number of output channels, the number of input channels,
    height and width of the kernels, respectively.
    The filter weight is initialized with i.i.d. Gaussian random samples, each
    of which has zero mean and deviation :math:`\sqrt{1/(c_I k_H k_W)}` by
    default. The deviation is scaled by ``wscale`` if specified.

    The bias vector is of size :math:`c_O`.
    Each element of it is initialized by ``bias`` argument.
    If ``nobias`` argument is set to True, then this function does not hold
    the bias parameter.

    The two-dimensional convolution function is defined as follows.
    Let :math:`X` be the input tensor of dimensions :math:`(n, c_I, h, w)`,
    where :math:`n` is the batch size, and :math:`(h, w)` is spatial size of
    the input image.
    Then the ``Convolution2D`` function computes correlations between filters
    and patches of size :math:`(k_H, k_W)` in :math:`X`.
    Note that correlation here is equivalent to the inner product between
    expanded vectors.
    Patches are extracted at positions shifted by multiples of ``stride`` from
    the first position ``-pad`` for each spatial axis.
    The right-most (or bottom-most) patches do not run over the padded spatial
    size.

    Let :math:`(s_Y, s_X)` be the stride of filter application, and
    :math:`(p_H, p_W)` the spatial padding size. Then, the output size
    :math:`(h_O, w_O)` is determined by the following equations:

    .. math::

       h_O &= (h + 2p_H - k_H) / s_Y + 1,\\\\
       w_O &= (w + 2p_W - k_W) / s_X + 1.

    """
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False, use_cudnn=True,
                 initialW=None, initial_bias=None,
                 dtype=numpy.float32):
        super(Convolution2D, self).__init__()
        self.dtype = numpy.dtype(dtype)

        ksize = _pair(ksize)
        stride = _pair(stride)
        pad = _pair(pad)

        self.kh, self.kw = ksize
        self.sy, self.sx = stride
        self.ph, self.pw = pad

        self.in_channels = in_channels
        self.out_channels = out_channels

        if initialW is not None:
            assert initialW.shape == \
                (out_channels, in_channels, self.kh, self.kw)
            self.params['W'] = initialW
        else:
            self.params['W'] = numpy.random.normal(
                0, wscale * math.sqrt(1. / (self.kh * self.kw * in_channels)),
                (out_channels, in_channels, self.kh, self.kw)
            ).astype(self.dtype)
        xp = cuda.get_array_module(self.params['W'])

        if initial_bias is not None:
            assert initial_bias.shape == (out_channels,)
            self.params['b'] = initial_bias
        elif not nobias:
            self.params['b'] = numpy.repeat(self.dtype.type(bias),
                                            out_channels)

        self.use_cudnn = use_cudnn
        if cuda.cudnn_enabled and use_cudnn:
            # chance to choose implicit-precomp-gemm algorithm
            self.max_workspace_size = in_channels * self.kh * self.kw * 4

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == self.dtype,
            x_type.ndim == 4,
            x_type.shape[1] == self.in_channels
        )

    def forward_cpu(self, x):
        self.col = conv.im2col_cpu(
            x[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw)
        y = numpy.tensordot(self.col, self.params['W'], ((1, 2, 3), (1, 2, 3)))
        b = self.params.get('b', None)
        if b is not None:
            y += b
        return numpy.rollaxis(y, 3, 1),

    def forward_gpu(self, x):
        n, c, h, w = x[0].shape
        W = self.params['W']
        b = self.params.get('b', None)

        out_h = conv.get_conv_outsize(h, self.kh, self.sy, self.ph)
        out_w = conv.get_conv_outsize(w, self.kw, self.sx, self.pw)
        out_c = W.shape[0]

        y = cuda.empty((n, out_c, out_h, out_w), dtype=self.dtype)
        if cuda.cudnn_enabled and self.use_cudnn:
            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x[0])
            y_desc = cudnn.create_tensor_descriptor(y)

            self.filter_desc = cudnn.create_filter_descriptor(self.W)
            self.conv_desc = cudnn.create_convolution_descriptor(
                (self.ph, self.pw), (self.sy, self.sx))
            if b is not None:
                self.bias_desc = cudnn.create_tensor_descriptor(
                    b[None, :, None, None])

            algo = libcudnn.getConvolutionForwardAlgorithm(
                handle, x_desc.value, self.filter_desc.value,
                self.conv_desc.value, y_desc.value, _fwd_pref,
                self.max_workspace_size)
            workspace_size = libcudnn.getConvolutionForwardWorkspaceSize(
                handle, x_desc.value, self.filter_desc.value,
                self.conv_desc.value, y_desc.value, algo)
            workspace = cuda.empty(
                (max(workspace_size // 4, 1),), dtype=self.dtype)

            one = ctypes.c_float(1)
            zero = ctypes.c_float(0)
            libcudnn.convolutionForward(
                handle, one, x_desc.value, x[0].data.ptr,
                self.filter_desc.value, self.W.data.ptr, self.conv_desc.value,
                algo, workspace.data.ptr, workspace_size, zero, y_desc.value,
                y.data.ptr)

            # TODO(beam2d): Support unshared bias
            if b is not None:
                libcudnn.addTensor(
                    handle, libcudnn.CUDNN_ADD_SAME_C, one,
                    self.bias_desc.value, b.data.ptr, one, y_desc.value,
                    y.data.ptr)
        else:
            # Implementation using im2col
            self.col = conv.im2col_gpu(
                x[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw)

            # TODO(beam2d): Use streams
            W_mat = W.reshape(out_c, c * self.kh * self.kw)
            col_mats = self.col.reshape(
                n, c * self.kh * self.kw, out_h * out_w)
            y_mats = y.reshape(n, out_c, out_h * out_w)
            for i in moves.range(n):
                y_mats[i] = W_mat.dot(col_mats[i])

            # TODO(beam2d): Support unshared bias
            if b is not None:
                y += b.reshape((1, out_c, 1, 1))

        return y,

    def backward_cpu(self, x, gy):
        if 'b' in self.params:
            self.grads['b'] += gy[0].sum(axis=(0, 2, 3))
        self.grads['W'] += numpy.tensordot(
            gy[0], self.col, ((0, 2, 3), (0, 4, 5)))
        gcol = numpy.tensordot(self.params['W'], gy[0], (0, 1))
        gcol = numpy.rollaxis(gcol, 3)

        h, w = x[0].shape[2:]
        return conv.col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w),

    def backward_gpu(self, x, gy):
        out_c, out_h, out_w = gy[0].shape[1:]
        n, c, h, w = x[0].shape

        if cuda.cudnn_enabled and self.use_cudnn:
            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x[0])
            gy_arr = gy[0]
            if not gy_arr.flags.c_contiguous:
                gy_arr = cuda.cupy.ascontiguousarray(gy_arr)
            gy_desc = cudnn.create_tensor_descriptor(gy_arr)
            one = ctypes.c_float(1)
            zero = ctypes.c_float(0)
            if 'b' in self.params:
                libcudnn.convolutionBackwardBias(
                    handle, one, gy_desc.value, gy_arr.data.ptr,
                    one, self.bias_desc.value, self.grads['b'].data.ptr)

            libcudnn.convolutionBackwardFilter(
                handle, one, x_desc.value, x[0].data.ptr,
                gy_desc.value, gy_arr.data.ptr, self.conv_desc.value,
                one, self.filter_desc.value, self.grads['W'].data.ptr)

            gx = cuda.empty_like(x[0])
            libcudnn.convolutionBackwardData(
                handle, one, self.filter_desc.value, self.params['W'].data.ptr,
                gy_desc.value, gy_arr.data.ptr, self.conv_desc.value,
                zero, x_desc.value, gx.data.ptr)
        else:
            if 'b' in self.params:
                self.gb += gy[0].sum(axis=(0, 2, 3))

            # TODO(beam2d): Use streams
            gW_mat = self.grads['W'].reshape(out_c, c * self.kh * self.kw)
            col_mats = self.col.reshape(
                n, c * self.kh * self.kw, out_h * out_w)
            gy_mats = gy[0].reshape(n, out_c, out_h * out_w)
            for i in moves.range(n):
                gW_mat += cuda.cupy.dot(gy_mats[i], col_mats[i].T)

            W_mat = self.params['W'].reshape(out_c, c * self.kh * self.kw)
            gcol = cuda.empty_like(self.col)
            gcol_mats = gcol.reshape(n, c * self.kh * self.kw, out_h * out_w)
            for i in moves.range(n):
                cuda.cupy.dot(W_mat.T, gy_mats[i], gcol_mats[i])

            gx = conv.col2im_gpu(
                gcol, self.sy, self.sx, self.ph, self.pw, h, w)

        return gx,


class NonparameterizedConvolution2D(function.Function):

    """Two-dimensional nonparameterized convolution class.

    Args:
        stride (int or (int, int)): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or (int, int)): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        use_cudnn (bool): If True, then this function uses CuDNN if available.

    .. seealso:: :class:`Convolution2D`

    """
    def __init__(self, stride=1, pad=0, use_cudnn=True):
        self.stride = stride
        self.pad = pad

        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(
            2 <= in_types.size(),
            in_types.size() <= 3,
        )

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype == numpy.float32,
            w_type.dtype == numpy.float32,
            x_type.ndim == 4,
            w_type.ndim == 4,
            x_type.shape[1] == w_type.shape[1],
        )

        if in_types.size().eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == numpy.float32,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward(self, x):
        W = x[1]
        if len(x) == 3:
            func = Convolution2D(
                W.shape[1], W.shape[0], W.shape[2:],
                stride=self.stride, pad=self.pad, use_cudnn=self.use_cudnn,
                initialW=W, initial_bias=x[2])
        else:
            func = Convolution2D(
                W.shape[1], W.shape[0], W.shape[2:],
                stride=self.stride, pad=self.pad, use_cudnn=self.use_cudnn,
                initialW=W, nobias=True)
        self.func = func
        if any(isinstance(i, cuda.ndarray) for i in x):
            func.to_gpu()
        return func.forward(x[:1])

    def backward(self, x, gy):
        func = self.func
        func.zerograds()
        gx = func.backward(x[:1], gy)
        gb = func.grads.get('b', None)
        if gb is None:
            return (gx[0], func.grads['W'])
        return (gx[0], func.grads['W'], gb)


def convolution_2d(x, W, b=None, stride=1, pad=0, use_cudnn=True):
    """Two-dimensional convolution function.

    Args:
        x (~chainer.Variable): Input variable.
        W (~chainer.Variable): Weight variable.
        b (~chainer.Variable): Bias  variable (optional).
        stride (int or (int, int)): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or (int, int)): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        use_cudnn (bool): If True, then this function uses CuDNN if available.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`Convolution2D`

    """
    if b is None:
        return NonparameterizedConvolution2D(
            stride=stride, pad=pad, use_cudnn=use_cudnn)(x, W)
    else:
        return NonparameterizedConvolution2D(
            stride=stride, pad=pad, use_cudnn=use_cudnn)(x, W, b)
