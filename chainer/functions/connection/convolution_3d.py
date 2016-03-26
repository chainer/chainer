import numpy 
from six import moves

from chainer import cuda
from chainer import function
from chainer.utils import conv
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _fwd_pref = libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
    if _cudnn_version >= 4000:
        _bwd_filter_pref = \
            libcudnn.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
        _bwd_data_pref = \
            libcudnn.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT


def _triplet(x):
    if hasattr(x, '__getitem__'):
        return x
    return (x, x, x)


def im32col_cpu(img, kh, kw, kd, sy, sx, sz, ph, pw, pd, pval=0, cover_all=False):
    n, c, h, w, d = img.shape
    out_h = conv.get_conv_outsize(h, kh, sy, ph, cover_all)
    out_w = conv.get_conv_outsize(w, kw, sx, pw, cover_all)
    out_d = conv.get_conv_outsize(d, kd, sz, pd, cover_all)
    img = numpy.pad(img,
                    ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1), (pd, pd + sz - 1)),
                    mode='constant', constant_values=(pval,))
    col = numpy.ndarray((n, c, kh, kw, kd, out_h, out_w, out_d), dtype=img.dtype)

    for i in moves.range(kh):
        i_lim = i + sy * out_h
        for j in moves.range(kw):
            j_lim = j + sx * out_w
            for k in moves.range(kd):
                k_lim = k + sz * out_d
                col[:, :, i, j, k, :, :, :] = img[:, :, i:i_lim:sy, j:j_lim:sx, k:k_lim:sz]

    return col


def col2im3_cpu(col, sy, sx, sz, ph, pw, pd, h, w, d):
    n, c, kh, kw, kd, out_h, out_w, out_d = col.shape
    img = numpy.zeros((n, c, h + 2 * ph + sy - 1, w + 2 * pw + sx - 1, d + 2 * pd + sz - 1),
                      dtype=col.dtype)
    for i in moves.range(kh):
        i_lim = i + sy * out_h
        for j in moves.range(kw):
            j_lim = j + sx * out_w
            for k in moves.range(kd):
                k_lim = k + sz * out_d
                img[:, :, i:i_lim:sy, j:j_lim:sx, k:k_lim:sz] += col[:, :, i, j, k, :, :, :]

    return img[:, :, ph:h + ph, pw:w + pw, pd:d + pd]


class Convolution3DFunction(function.Function):

    def __init__(self, stride=1, pad=0, use_cudnn=True):
        self.sy, self.sx, self.sz = _triplet(stride)
        self.ph, self.pw, self.pd = _triplet(pad)
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype == numpy.float32,
            w_type.dtype == numpy.float32,
            x_type.ndim == 5,
            w_type.ndim == 5,
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
        b = inputs[2] if len(inputs) == 3 else None
        kh, kw, kd = W.shape[2:]
        self.col = im32col_cpu(
            x, kh, kw, kd, self.sy, self.sx, self.sz, self.ph, self.pw, self.pd)
        y = numpy.tensordot(self.col, W, ((1, 2, 3, 4), (1, 2, 3, 4)))
        if b is not None:
            y += b
        return numpy.rollaxis(y, 4, 1),


    def forward_gpu(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        out_c, _, kh, kw, kd  = W.shape
        n, c, h, w, d = x.shape
        out_h = conv.get_conv_outsize(h, kh, self.sy, self.ph)
        out_w = conv.get_conv_outsize(w, kw, self.sx, self.pw)
        out_d = conv.get_conv_outsize(d, kd, self.sz, self.pd)

        y = cuda.cupy.empty((n, out_c, out_h, out_w, out_d), dtype=x.dtype)
        if cuda.cudnn_enabled and self.use_cudnn:
            x = cuda.cupy.ascontiguousarray(x)
            W = cuda.cupy.ascontiguousarray(W)
            if b is not None:
                b = cuda.cupy.ascontiguousarray(b)

            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x)
            y_desc = cudnn.create_tensor_descriptor(y)

            self.filter_desc = cudnn.create_filter_descriptor(W)
            self.conv_desc = cudnn.create_convolution_descriptor(
                (self.ph, self.pw, self.pd), (self.sy, self.sx, self.sz))
            if b is not None:
                self.bias_desc = cudnn.create_tensor_descriptor(
                    b[None, :, None, None, None])

            self.max_workspace_size = c * kh * kw * kd * 4
            algo = libcudnn.getConvolutionForwardAlgorithm(
                handle, x_desc.value, self.filter_desc.value,
                self.conv_desc.value, y_desc.value, _fwd_pref,
                self.max_workspace_size)
            workspace_size = libcudnn.getConvolutionForwardWorkspaceSize(
                handle, x_desc.value, self.filter_desc.value,
                self.conv_desc.value, y_desc.value, algo)
            workspace = cuda.cupy.empty(
                (max(workspace_size // 4, 1),), dtype=x.dtype)

            dtype = x.dtype
            one = numpy.array(1, dtype=dtype).ctypes
            zero = numpy.array(0, dtype=dtype).ctypes
            libcudnn.convolutionForward(
                handle, one.data, x_desc.value, x.data.ptr,
                self.filter_desc.value, W.data.ptr, self.conv_desc.value,
                algo, workspace.data.ptr, workspace_size, zero.data,
                y_desc.value, y.data.ptr)

            # TODO(beam2d): Support unshared bias
            if b is not None:
                libcudnn.addTensor(
                    handle, one.data,
                    self.bias_desc.value, b.data.ptr, one.data,
                    y_desc.value, y.data.ptr)
        else:
            raise NotImplementedError('forward_gpu without cudnn is not implemented yet.')

        return y,



    def backward_cpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        h, w, d = x.shape[2:]

        gW = numpy.tensordot(gy, self.col, ((0, 2, 3, 4), (0, 5, 6, 7)))
        gcol = numpy.tensordot(W, gy, (0, 1))
        gcol = numpy.rollaxis(gcol, 4)
        gx = col2im3_cpu(gcol, self.sy, self.sx, self.sz, self.ph, self.pw, self.pd, h, w, d)

        if b is None:
            return gx, gW
        else:
            gb = gy.sum(axis=(0, 2, 3, 4))
            return gx, gW, gb


    def backward_gpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        _, out_c, out_h, out_w, out_d = gy.shape
        n, c, h, w, d = x.shape
        kh, kw, kd = W.shape[2:]

        gW = cuda.cupy.empty_like(W)
        if cuda.cudnn_enabled and self.use_cudnn:
            x = cuda.cupy.ascontiguousarray(x)
            W = cuda.cupy.ascontiguousarray(W)
            gy = cuda.cupy.ascontiguousarray(gy)

            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(x)
            gy_desc = cudnn.create_tensor_descriptor(gy)
            dtype = x.dtype
            one = numpy.array(1, dtype=dtype).ctypes
            zero = numpy.array(0, dtype=dtype).ctypes

            if _cudnn_version >= 4000:
                self.max_workspace_size = c * kh * kw * kd * 4
                algo = libcudnn.getConvolutionBackwardFilterAlgorithm(
                    handle, x_desc.value, gy_desc.value,
                    self.conv_desc.value, self.filter_desc.value,
                    _bwd_filter_pref, self.max_workspace_size)
                workspace_size = \
                    libcudnn.getConvolutionBackwardFilterWorkspaceSize(
                        handle, x_desc.value, gy_desc.value,
                        self.conv_desc.value, self.filter_desc.value, algo)
                workspace = cuda.cupy.empty(
                    (max(workspace_size // 4, 1),), dtype=x.dtype)

                libcudnn.convolutionBackwardFilter_v3(
                    handle, one.data, x_desc.value, x.data.ptr,
                    gy_desc.value, gy.data.ptr, self.conv_desc.value,
                    algo, workspace.data.ptr, workspace_size,
                    zero.data, self.filter_desc.value, gW.data.ptr)
            else:
                libcudnn.convolutionBackwardFilter_v2(
                    handle, one.data, x_desc.value, x.data.ptr,
                    gy_desc.value, gy.data.ptr, self.conv_desc.value,
                    zero.data, self.filter_desc.value, gW.data.ptr)

            gx = cuda.cupy.empty_like(x)

            if _cudnn_version >= 4000:
                algo = libcudnn.getConvolutionBackwardDataAlgorithm(
                    handle, self.filter_desc.value, gy_desc.value,
                    self.conv_desc.value, x_desc.value, _bwd_data_pref,
                    self.max_workspace_size)
                workspace_size = \
                    libcudnn.getConvolutionBackwardDataWorkspaceSize(
                        handle, self.filter_desc.value, gy_desc.value,
                        self.conv_desc.value, x_desc.value, algo)
                workspace = cuda.cupy.empty(
                    (max(workspace_size // 4, 1),), dtype=x.dtype)

                libcudnn.convolutionBackwardData_v3(
                    handle, one.data, self.filter_desc.value, W.data.ptr,
                    gy_desc.value, gy.data.ptr, self.conv_desc.value,
                    algo, workspace.data.ptr, workspace_size,
                    zero.data, x_desc.value, gx.data.ptr)
            else:
                libcudnn.convolutionBackwardData_v2(
                    handle, one.data, self.filter_desc.value, W.data.ptr,
                    gy_desc.value, gy.data.ptr, self.conv_desc.value,
                    zero.data, x_desc.value, gx.data.ptr)

            if b is not None:
                gb = cuda.cupy.empty_like(b)
                libcudnn.convolutionBackwardBias(
                    handle, one.data, gy_desc.value, gy.data.ptr,
                    zero.data, self.bias_desc.value, gb.data.ptr)
        else:
            raise NotImplementedError('forward_gpu without cudnn is not implemented yet.')            

        if b is None:
            return gx, gW
        else:
            return gx, gW, gb


def convolution_3d(x, W, b=None, stride=1, pad=0, use_cudnn=True):
    """Three-dimensional convolution function.
    """
    func = Convolution3DFunction(stride, pad, use_cudnn)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)





