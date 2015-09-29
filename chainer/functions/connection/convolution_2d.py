import ctypes
import math

import numpy
from six import moves

from chainer import cuda
from chainer import function
from chainer import link
from chainer.utils import conv
from chainer.utils import type_check
from chainer import variable

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn
    _fwd_pref = libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return (x, x)


class Convolution2DFunction(function.Function):

    def __init__(self, stride=1, pad=0, use_cudnn=True):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype == numpy.float32,
            w_type.dtype == numpy.float32,
            x_type.ndim == 4,
            w_type.ndim == 4,
            x_type.shape[1] == w_type.shape[1],
        )

        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == numpy.float32,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward_cpu(self, inputs):
        x, W = inputs[:2]
        kh, kw = W.shape[2:]
        self.col = conv.im2col_cpu(
            x, kh, kw, self.sy, self.sx, self.ph, self.pw)
        y = numpy.tensordot(self.col, W, ((1, 2, 3), (1, 2, 3)))
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return numpy.rollaxis(y, 3, 1),

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, W = inputs[:2]
        out_c, _, kh, kw = W.shape

        n, c, h, w = x.shape
        b = inputs[2] if len(inputs) == 3 else None

        out_h = conv.get_conv_outsize(h, kh, self.sy, self.ph)
        out_w = conv.get_conv_outsize(w, kw, self.sx, self.pw)

        y = cupy.empty((n, out_c, out_h, out_w), dtype=x.dtype)
        if cuda.cudnn_enabled and self.use_cudnn:
            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x)
            y_desc = cudnn.create_tensor_descriptor(y)

            self.filter_desc = cudnn.create_filter_descriptor(W)
            self.conv_desc = cudnn.create_convolution_descriptor(
                (self.ph, self.pw), (self.sy, self.sx))
            if b is not None:
                self.bias_desc = cudnn.create_tensor_descriptor(
                    b[None, :, None, None])

            self.max_workspace_size = c * kh * kw * 4
            algo = libcudnn.getConvolutionForwardAlgorithm(
                handle, x_desc.value, self.filter_desc.value,
                self.conv_desc.value, y_desc.value, _fwd_pref,
                self.max_workspace_size)
            workspace_size = libcudnn.getConvolutionForwardWorkspaceSize(
                handle, x_desc.value, self.filter_desc.value,
                self.conv_desc.value, y_desc.value, algo)
            workspace = cupy.empty(
                (max(workspace_size // 4, 1),), dtype=x.dtype)

            one = ctypes.c_float(1)
            zero = ctypes.c_float(0)
            libcudnn.convolutionForward(
                handle, one, x_desc.value, x.data.ptr,
                self.filter_desc.value, W.data.ptr,
                self.conv_desc.value, algo, workspace.data.ptr, workspace_size,
                zero, y_desc.value, y.data.ptr)

            # TODO(beam2d): Support unshared bias
            if b is not None:
                libcudnn.addTensor(
                    handle, libcudnn.CUDNN_ADD_SAME_C, one,
                    self.bias_desc.value, b.data.ptr, one, y_desc.value,
                    y.data.ptr)
        else:
            # Implementation using im2col
            self.col = conv.im2col_gpu(
                x, kh, kw, self.sy, self.sx, self.ph, self.pw)
            W_mat = W.reshape(out_c, -1)
            col_mats = self.col.reshape(n, -1, out_h * out_w)
            y_mats = y.reshape(n, out_c, -1)
            # TODO(beam2d): Use streams or batch gemm
            for i in moves.range(n):
                y_mats[i] = W_mat.dot(col_mats[i])
            # TODO(beam2d): Support unshared bias
            if b is not None:
                y += b[:, None, None]

        return y,

    def backward_cpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        gy = grad_outputs[0]
        h, w = x.shape[2:]

        gW = numpy.tensordot(gy, self.col, ((0, 2, 3), (0, 4, 5)))
        gcol = numpy.tensordot(W, gy, (0, 1))
        gcol = numpy.rollaxis(gcol, 3)
        gx = conv.col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w)

        if len(inputs) == 3:
            gb = gy.sum(axis=(0, 2, 3))
            return gx, gW, gb
        else:
            return gx, gW

    def backward_gpu(self, inputs, grad_outputs):
        cupy = cuda.cupy
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        _, out_c, out_h, out_w = gy.shape
        n, c, h, w = x.shape
        kh, kw = W.shape[2:]

        gW = cupy.empty_like(W)
        if cuda.cudnn_enabled and self.use_cudnn:
            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x)
            if not gy.flags.c_contiguous:
                gy = cupy.ascontiguousarray(gy)
            gy_desc = cudnn.create_tensor_descriptor(gy)
            one = ctypes.c_float(1)
            zero = ctypes.c_float(0)

            libcudnn.convolutionBackwardFilter(
                handle, one, x_desc.value, x.data.ptr,
                gy_desc.value, gy.data.ptr, self.conv_desc.value,
                zero, self.filter_desc.value, gW.data.ptr)

            gx = cupy.empty_like(x)
            libcudnn.convolutionBackwardData(
                handle, one, self.filter_desc.value, W.data.ptr,
                gy_desc.value, gy.data.ptr, self.conv_desc.value,
                zero, x_desc.value, gx.data.ptr)

            if b is not None:
                gb = cupy.empty_like(inputs[2])
                libcudnn.convolutionBackwardBias(
                    handle, one, gy_desc.value, gy.data.ptr,
                    zero, self.bias_desc.value, gb.data.ptr)
        else:
            gW_mat = gW.reshape(out_c, c * self.kh * self.kw)
            col_mats = self.col.reshape(n, c * kh * kw, out_h * out_w)
            gy_mats = gy.reshape(n, out_c, out_h * out_w)
            # TODO(beam2d): Use streams or batch gemm
            for i in moves.range(n):
                gW_mat += cupy.dot(gy_mats[i], col_mats[i].T)

            W_mat = W.reshape(out_c, -1)
            gcol = cupy.empty_like(self.col)
            gcol_mats = gcol.reshape(n, c * kh * kw, out_h * out_w)
            for i in moves.range(n):
                cupy.dot(W_mat.T, gy_mats[i], gcol_mats[i])

            gx = conv.col2im_gpu(
                gcol, self.sy, self.sx, self.ph, self.pw, h, w)

            if b is not None:
                gb = gy.sum(axis=(0, 2, 3))

        if b is None:
            return gx, gW
        else:
            return gx, gW, gb


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
    func = Convolution2DFunction(stride, pad, use_cudnn)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)


class Convolution2D(link.Link):

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
        self._conv_arg = (stride, pad, use_cudnn)

        dtype = numpy.dtype(dtype)
        kh, kw = _pair(ksize)

        W_shape = (out_channels, in_channels, kh, kw)
        if initialW is not None:
            assert initialW.shape == W_shape
            self.params['W'] = variable.Variable(initialW)
        else:
            std = wscale * math.sqrt(1. / (kh * kw * in_channels))
            self.params['W'] = variable.Variable(numpy.random.normal(
                0, std, W_shape).astype(dtype))

        if initial_bias is not None:
            assert initial_bias.shape == (out_channels,)
            self.params['b'] = variable.Variable(initial_bias)
        elif not nobias:
            self.params['b'] = variable.Variable(
                numpy.repeat(dtype.type(bias), out_channels))

    def __call__(self, x):
        W = self.params['W']
        b = self.params.get('b', None)
        return convolution_2d(x, W, b, *self._conv_arg)
