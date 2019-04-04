import numpy

import chainer
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import function_node
from chainer.functions.pooling import pooling_2d
from chainer.utils import conv
import chainerx


class MaxPooling2D(pooling_2d.Pooling2D):

    """Max pooling over a set of 2d planes."""

    def forward_chainerx(self, x):
        # TODO(sonots): Support return_indices in ChainerX
        if self.return_indices:
            return chainer.Fallback
        return chainerx.max_pool(x[0], (self.kh, self.kw), (self.sy, self.sx),
                                 (self.ph, self.pw), self.cover_all),

    def forward_cpu(self, x):
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(x)):
            return self._forward_ideep(x)

        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype

        col = conv.im2col_cpu(
            x[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            pval=-float('inf'), cover_all=self.cover_all)
        n, c, kh, kw, out_h, out_w = col.shape
        col = col.reshape(n, c, kh * kw, out_h, out_w)

        # We select maximum twice, since the implementation using numpy.choose
        # hits its bug when kh * kw >= 32.
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y,

    def _forward_ideep(self, x):
        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype
        self.retain_inputs((0,))

        n, c, h, w = x[0].shape
        y_h = conv.get_conv_outsize(
            h, self.kh, self.sy, self.ph, self.cover_all)
        assert y_h > 0, 'Height in the output should be positive.'
        y_w = conv.get_conv_outsize(
            w, self.kw, self.sx, self.pw, self.cover_all)
        assert y_w > 0, 'Width in the output should be positive.'
        self.pd = self.sy * (y_h - 1) + self.kh - h - self.ph
        self.pr = self.sx * (y_w - 1) + self.kw - w - self.pw

        pp = intel64.ideep.pooling2DParam(
            (n, c, y_h, y_w),
            self.kh, self.kw,
            self.sy, self.sx,
            self.ph, self.pw,
            self.pd, self.pr,
            intel64.ideep.pooling2DParam.pooling_max)
        y, self.indexes = intel64.ideep.pooling2D.Forward(
            intel64.ideep.array(x[0]), pp)
        return y,

    def forward_gpu(self, x):
        if chainer.should_use_cudnn('>=auto'):
            return super(MaxPooling2D, self).forward_gpu(x)

        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype

        n, c, h, w = x[0].shape
        y_h = conv.get_conv_outsize(
            h, self.kh, self.sy, self.ph, self.cover_all)
        assert y_h > 0, 'Height in the output should be positive.'
        y_w = conv.get_conv_outsize(
            w, self.kw, self.sx, self.pw, self.cover_all)
        assert y_w > 0, 'Width in the output should be positive.'
        y = cuda.cupy.empty((n, c, y_h, y_w), dtype=x[0].dtype)
        self.indexes = cuda.cupy.empty((n, c, y_h, y_w), dtype=numpy.int32)

        cuda.elementwise(
            'raw T in, int32 h, int32 w, int32 out_h, int32 out_w,'
            'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw',
            'T out, S indexes',
            '''
               int c0    = i / (out_h * out_w);
               int out_y = i / out_w % out_h;
               int out_x = i % out_w;
               int in_y_0 = max(0, out_y * sy - ph);
               int in_y_1 = min(h, out_y * sy + kh - ph);
               int in_x_0 = max(0, out_x * sx - pw);
               int in_x_1 = min(w, out_x * sx + kw - pw);

               T maxval = in[in_x_0 + w * (in_y_0 + h * c0)];
               int argmax_y = in_y_0;
               int argmax_x = in_x_0;
               for (int y = in_y_0; y < in_y_1; ++y) {
                 int offset_y = w * (y + h * c0);
                 for (int x = in_x_0; x < in_x_1; ++x) {
                   float v = in[x + offset_y];
                   if (maxval < v) {
                     maxval   = v;
                     argmax_y = y;
                     argmax_x = x;
                   }
                 }
               }
               out = maxval;

               int argmax_ky = argmax_y + ph - out_y * sy;
               int argmax_kx = argmax_x + pw - out_x * sx;
               indexes = argmax_kx + kw * argmax_ky;
            ''', 'max_pool_fwd')(x[0].reduced_view(),
                                 h, w, y_h, y_w, self.kh, self.kw,
                                 self.sy, self.sx, self.ph, self.pw,
                                 y, self.indexes)
        return y,

    def backward(self, indexes, gy):
        return MaxPooling2DGrad(self).apply(gy)

    def _get_pool_mode(self):
        return cuda.cuda.cudnn.CUDNN_POOLING_MAX


class MaxPooling2DGrad(function_node.FunctionNode):

    def __init__(self, mpool2d):
        self.kh = mpool2d.kh
        self.kw = mpool2d.kw
        self.sy = mpool2d.sy
        self.sx = mpool2d.sx
        self.ph = mpool2d.ph
        self.pw = mpool2d.pw
        self.cover_all = mpool2d.cover_all
        self._used_cudnn = mpool2d._used_cudnn
        if not self._used_cudnn:
            self.indexes = mpool2d.indexes
            self._in_shape = mpool2d._in_shape
            self._in_dtype = mpool2d._in_dtype
        self.mpool2d = mpool2d

    def forward_cpu(self, gy):
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(gy)):
            return self._forward_ideep(gy)

        n, c, out_h, out_w = gy[0].shape
        h, w = self._in_shape[2:]
        kh, kw = self.kh, self.kw

        gcol = numpy.zeros(
            (n * c * out_h * out_w * kh * kw), dtype=self._in_dtype)

        indexes = self.indexes.flatten()
        indexes += numpy.arange(0, indexes.size * kh * kw, kh * kw)

        gcol[indexes] = gy[0].ravel()
        gcol = gcol.reshape(n, c, out_h, out_w, kh, kw)
        gcol = numpy.swapaxes(gcol, 2, 4)
        gcol = numpy.swapaxes(gcol, 3, 5)

        gx = conv.col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w)
        return gx,

    def _forward_ideep(self, gy):
        # FIXME
        # Here we expect indexes is returned from MKL-DNN
        # otherwise, there are dtype mismatch for reorder (int64-->uint8)
        if not isinstance(self.indexes, intel64.ideep.mdarray):
            return self.forward_cpu(gy)

        n, c, h, w = self._in_shape
        y_h, y_w = gy[0].shape[2:]
        x = self.mpool2d.get_retained_inputs()[0].array

        self.pd = self.sy * (y_h - 1) + self.kh - h - self.ph
        self.pr = self.sx * (y_w - 1) + self.kw - w - self.pw

        pp = intel64.ideep.pooling2DParam(
            self._in_shape,
            self.kh, self.kw,
            self.sy, self.sx,
            self.ph, self.pw,
            self.pd, self.pr,
            intel64.ideep.pooling2DParam.pooling_max)

        self.indexes = intel64.ideep.array(self.indexes)
        gx = intel64.ideep.pooling2D.Backward(
            intel64.ideep.array(x),
            intel64.ideep.array(gy[0]),
            self.indexes, pp)
        return gx,

    def forward_gpu(self, gy):
        if self._used_cudnn:
            x = self.mpool2d.get_retained_inputs()[0].array
            return self.mpool2d.backward_gpu((x,), gy)
        n, c, h, w = self._in_shape
        y_h, y_w = gy[0].shape[2:]
        gx = cuda.cupy.empty(self._in_shape, self._in_dtype)

        cuda.elementwise(
            'raw T gy, raw S indexes, int32 h, int32 w,'
            'int32 out_h, int32 out_w, int32 kh, int32 kw,'
            'int32 sy, int32 sx, int32 ph, int32 pw',
            'T gx',
            '''
               int c0 = i / (h * w);
               int y  = i / w % h + ph;
               int x  = i % w + pw;
               int out_y_0 = max(0,     (y - kh + sy) / sy);
               int out_y_1 = min(out_h, (y      + sy) / sy);
               int out_x_0 = max(0,     (x - kw + sx) / sx);
               int out_x_1 = min(out_w, (x      + sx) / sx);

               T val = 0;
               for (int out_y = out_y_0; out_y < out_y_1; ++out_y) {
                 int ky = y - out_y * sy;
                 for (int out_x = out_x_0; out_x < out_x_1; ++out_x) {
                   int kx = x - out_x * sx;
                   int offset = out_x + out_w * (out_y + out_h * c0);
                   if (indexes[offset] == kx + kw * ky) {
                     val = val + gy[offset];
                   }
                 }
               }
               gx = val;
            ''',
            'max_pool_bwd')(gy[0].reduced_view(), self.indexes.reduced_view(),
                            h, w, y_h, y_w, self.kh, self.kw,
                            self.sy, self.sx, self.ph, self.pw,
                            gx)
        return gx,

    def backward(self, indexes, ggx):
        return MaxPooling2DWithIndexes(self.mpool2d).apply(ggx)


class MaxPooling2DWithIndexes(function_node.FunctionNode):

    def __init__(self, mpool2d):
        self.kh = mpool2d.kh
        self.kw = mpool2d.kw
        self.sy = mpool2d.sy
        self.sx = mpool2d.sx
        self.ph = mpool2d.ph
        self.pw = mpool2d.pw
        self.cover_all = mpool2d.cover_all
        self._used_cudnn = mpool2d._used_cudnn
        if not self._used_cudnn:
            self.indexes = mpool2d.indexes
        else:
            self.mpool2d = mpool2d

    def forward_cpu(self, x):
        col = conv.im2col_cpu(
            x[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            pval=-float('inf'), cover_all=self.cover_all)
        n, c, kh, kw, out_h, out_w = col.shape
        col = col.reshape(n, c, kh * kw, out_h, out_w)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, kh * kw)
        indexes = self.indexes.ravel()
        col = col[numpy.arange(len(indexes)), indexes]
        return col.reshape(n, c, out_h, out_w),

    def forward_gpu(self, inputs):
        if self._used_cudnn:
            x = self.mpool2d.get_retained_inputs()[0].array
            return self._forward_gpu_compute_indexes_again((x, inputs[0]))
        else:
            x, = inputs
            n, c, h, w = x.shape
            y_h = conv.get_conv_outsize(
                h, self.kh, self.sy, self.ph, self.cover_all)
            assert y_h > 0, 'Height in the output should be positive.'
            y_w = conv.get_conv_outsize(
                w, self.kw, self.sx, self.pw, self.cover_all)
            assert y_w > 0, 'Width in the output should be positive.'
            y = cuda.cupy.empty((n, c, y_h, y_w), dtype=x.dtype)

            cuda.elementwise(
                'raw T in, raw S indexes, int32 h, int32 w, int32 out_h,'
                'int32 out_w, int32 kh, int32 kw, int32 sy, int32 sx,'
                'int32 ph, int32 pw', 'T out',
                '''
                int c0    = i / (out_h * out_w);
                int out_y = i / out_w % out_h;
                int out_x = i % out_w;
                int index = indexes[i];
                int max_y = max(0, out_y * sy - ph + index / kw);
                int max_x = max(0, out_x * sx - pw + index % kw);
                out = in[max_x + w * (max_y + h * c0)];
                ''', 'max_pool_grad_fwd')(
                    x.reduced_view(), self.indexes.reduced_view(), h, w,
                    y_h, y_w, self.kh, self.kw, self.sy, self.sx, self.ph,
                    self.pw, y)
            return y,

    def _forward_gpu_compute_indexes_again(self, inputs):
        x, ggx = inputs
        n, c, h, w = ggx.shape
        y_h = conv.get_conv_outsize(
            h, self.kh, self.sy, self.ph, self.cover_all)
        assert y_h > 0, 'Height in the output should be positive.'
        y_w = conv.get_conv_outsize(
            w, self.kw, self.sx, self.pw, self.cover_all)
        assert y_w > 0, 'Width in the output should be positive.'
        y = cuda.cupy.empty((n, c, y_h, y_w), dtype=x.dtype)

        cuda.elementwise(
            'raw T in, raw T ggx, int32 h, int32 w, int32 out_h,'
            'int32 out_w, int32 kh, int32 kw, int32 sy, int32 sx,'
            'int32 ph, int32 pw', 'T out',
            '''
            int c0    = i / (out_h * out_w);
            int out_y = i / out_w % out_h;
            int out_x = i % out_w;
            int in_y_0 = max(0, out_y * sy - ph);
            int in_y_1 = min(h, out_y * sy + kh - ph);
            int in_x_0 = max(0, out_x * sx - pw);
            int in_x_1 = min(w, out_x * sx + kw - pw);

            T maxval = in[in_x_0 + w * (in_y_0 + h * c0)];
            int argmax_y = in_y_0;
            int argmax_x = in_x_0;
            for (int y = in_y_0; y < in_y_1; ++y) {
                int offset_y = w * (y + h * c0);
                for (int x = in_x_0; x < in_x_1; ++x) {
                    float v = in[x + offset_y];
                    if (maxval < v) {
                        argmax_y = y;
                        argmax_x = x;
                    }
                }
            }
            out = ggx[argmax_x + w * (argmax_y + h * c0)]
            ''', 'max_pool_grad_fwd_calc_indexes')(
                x.reduced_view(), ggx.reduced_view(), h, w, y_h, y_w, self.kh,
                self.kw, self.sy, self.sx, self.ph, self.pw, y)
        return y,


def max_pooling_2d(x, ksize, stride=None, pad=0, cover_all=True,
                   return_indices=False):
    """Spatial max pooling function.

    This function acts similarly to :func:`~chainer.functions.convolution_2d`,
    but it computes the maximum of input spatial patch for each channel without
    any parameter instead of computing the inner products.

    Args:
        x (~chainer.Variable): Input variable.
        ksize (int or pair of ints): Size of pooling window. ``ksize=k`` and
            ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints or None): Stride of pooling applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent. If ``None`` is
            specified, then it uses same stride as the pooling window size.
        pad (int or pair of ints): Spatial padding width for the input array.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        cover_all (bool): If ``True``, all spatial locations are pooled into
            some output pixels. It may make the output size larger.
        return_indices (bool): If ``True``, pooling indices array is returned
            together with the output variable. The returned indices are
            expected for use by :func:`chainer.functions.upsampling_2d`.
            Note that cuDNN will not be used for this function if
            ``return_indices`` is set to ``True``, as cuDNN does not return
            indices information.

    Returns:
        ~chainer.Variable or tuple:
            When ``return_indices`` is ``False`` (default), returns the output
            variable.
            When ``True``, returns the tuple of the output variable and
            pooling indices (:ref:`ndarray`). Pooling indices will be on the
            same device as the input.

    """
    func = MaxPooling2D(ksize, stride, pad, cover_all, return_indices)
    if return_indices:
        with chainer.using_config('use_cudnn', 'never'):
            out = func.apply((x,))[0]
        return out, func.indexes

    return func.apply((x,))[0]
