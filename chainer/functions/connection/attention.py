import numpy
import six

from chainer import cuda
from chainer import function
from chainer.functions.math.matmul import _batch_matmul_gpu
from chainer.functions.math.matmul import _matmul
from chainer.utils import type_check
import cupy


def _split(inputs, pos):
    return inputs[:pos], inputs[pos:]


def _seqs_to_array(xs, length, pad_value):
    """concatenate variable length sequence to an array.

    gaps are filled by `pad_value`.

    """

    batchsize = len(xs)
    xp = cuda.get_array_module(*xs)
    dtype = xs[0].dtype
    unit = xs[0].shape[1:]
    outs = xp.full((batchsize, length) + unit, pad_value, dtype=dtype)

    if xp is numpy:
        for i, x in enumerate(xs):
            outs[i, :len(x), ...] = x

    else:
        offsets1 = numpy.empty(len(xs) + 1, dtype='i')
        offsets1[0] = 0
        numpy.cumsum([len(x) for x in xs], out=offsets1[1:])

        xsc = xp.concatenate(xs, axis=0)
        unit_size = xs[0].size // len(xs[0])
        size = length * batchsize * unit_size
        cuda.elementwise(
            'int32 len, int32 unit, raw int32 offsets1, raw T xsc',
            'raw T out',
            '''
            int ind = i / unit;
            int off = i - ind * unit;
            int y = ind / len;
            int x = ind - y * len;
            if (offsets1[y] + x < offsets1[y+1]){
              out[i] = xsc[(offsets1[y] + x) * unit + off];
            }
            ''',
            'seqs_to_array'
        )(length, unit_size, cuda.to_gpu(offsets1), xsc, outs, size=size)

    return outs


def _mask_array(array, lens, mask_value):
    """write mask values in place"""
    xp = cuda.get_array_module(array)

    if xp is numpy:
        for i, x in enumerate(array):
            array[i, lens[i]:] = mask_value

    else:
        maxlen = array.shape[1]
        lens = xp.array(lens).astype(xp.int32)

        cuda.elementwise(
            'T val, int32 len, raw int32 lens',
            'raw T array',
            '''
            int y = i / len;
            int x = i - y * len;
            if (lens[y] < x + 1){
              array[i] = val;
            }
            ''',
            'mask_array'
        )(mask_value, maxlen, cuda.to_gpu(lens), array, size=array.size)


class Attention(function.Function):

    def check_type_forward(self, in_types):
        q_type = in_types[0]
        x_types = in_types[1:]

        type_check.expect(
            q_type.dtype == numpy.float32,
            q_type.ndim == 2,
            q_type.shape[0] == len(x_types),
        )

        dim = q_type.shape[1]
        for x_type in x_types:
            type_check.expect(
                x_type.dtype == numpy.float32,
                x_type.ndim == 2,
                x_type.shape[1] == dim,
            )

    def forward_cpu(self, inputs):
        (q,), inputs = _split(inputs, 1)
        x_list = inputs
        dim = q.shape[1]
        batchsize = len(x_list)
        lens = map(len, x_list)
        max_len = max(lens)
        xmat = _seqs_to_array(x_list, max_len, 0)

        # inner product scores
        dtype = q.dtype
        score = numpy.empty((batchsize, max_len), dtype=dtype)
        for i in six.moves.range(batchsize):
            score[i] = _matmul(xmat[i], q[i], transa=False,
                               transb=False).ravel()

        # mask scores
        _mask_array(score, lens, -numpy.inf)

        # softmax
        alpha = score - score.max(axis=1, keepdims=True)
        numpy.exp(alpha, out=alpha)
        alpha /= alpha.sum(axis=1, keepdims=True)
        self.alpha = alpha

        # aggregate
        out = numpy.empty((batchsize, dim), dtype=dtype)
        for i in six.moves.range(batchsize):
            out[i] = _matmul(xmat[i], alpha[i], transa=True,
                             transb=False).ravel()
        self.out = out

        return out,

    def forward_gpu(self, inputs):
        (q,), inputs = _split(inputs, 1)
        x_list = inputs
        dim = q.shape[1]
        batchsize = len(x_list)
        lens = map(len, x_list)
        max_len = max(lens)
        xmat = _seqs_to_array(x_list, max_len, 0)

        score = cupy.empty((batchsize, max_len, 1), dtype=xmat.dtype)
        _batch_matmul_gpu(xmat, q, score)
        score = score.reshape((batchsize, max_len))
        _mask_array(score, lens, -numpy.inf)

        # softmax
        alpha = score - score.max(axis=1, keepdims=True)
        cupy.exp(alpha, out=alpha)
        alpha /= alpha.sum(axis=1, keepdims=True)
        self.alpha = alpha

        # aggregate
        out = cupy.empty((batchsize, dim, 1), dtype=xmat.dtype)
        _batch_matmul_gpu(xmat, alpha, out, transa=True)
        out = out.reshape((batchsize, dim))
        self.out = out

        return out,

    def backward_cpu(self, inputs, grads):
        q = inputs[0]
        x_list = inputs[1:]
        batchsize = len(x_list)
        lens = map(len, x_list)
        max_len = max(lens)
        xmat = _seqs_to_array(x_list, max_len, 0)

        gy, = grads

        # gq
        _a0 = numpy.sum(gy * self.out, axis=1)
        _b0 = numpy.sum(numpy.expand_dims(gy, axis=1) * xmat, axis=2)
        _b1 = numpy.expand_dims(_b0 * self.alpha, axis=2)
        gq = - numpy.expand_dims(_a0, axis=1) * \
            self.out + numpy.sum(_b1 * xmat, axis=1)

        # gx_list
        gx_list = []
        for i in range(batchsize):
            gyi = gy[i]
            xmati = xmat[i]
            alphai = self.alpha[i]
            outi = self.out[i]
            qi = q[i]
            _c = numpy.expand_dims(
                gyi, axis=0) * numpy.expand_dims(alphai, axis=1)
            _d0 = - numpy.dot(gyi, outi)
            _d = _d0 * numpy.expand_dims(alphai, axis=1) * \
                numpy.expand_dims(qi, axis=0)
            _e0 = numpy.tensordot(xmati, gyi, axes=([1], [0]))
            _e1 = _e0 * alphai
            _e = numpy.expand_dims(_e1, axis=1) * numpy.expand_dims(qi, axis=0)
            gx = _c + _d + _e
            gx_list.append(gx[:lens[i]])

        return tuple([gq, ] + gx_list)

    def backward_gpu(self, inputs, grads):
        q = inputs[0]
        x_list = inputs[1:]
        dim = q.shape[1]
        batchsize = len(x_list)
        lens = map(len, x_list)
        max_len = max(lens)
        xmat = _seqs_to_array(x_list, max_len, 0)
        dtype = q.dtype
        gy, = grads

        xp = cupy
        # gq
        _a0 = xp.sum(gy * self.out, axis=1)
        _a = xp.expand_dims(_a0, axis=1) * self.out
        _b0 = xp.empty((batchsize, max_len, 1), dtype=dtype)
        _batch_matmul_gpu(xmat, gy, _b0)
        _b1 = _b0 * xp.expand_dims(self.alpha, axis=2)
        _b2 = xp.empty((batchsize, 1, dim), dtype=dtype)
        _batch_matmul_gpu(_b1, xmat, _b2, transa=True)
        gq = - _a + _b2[:, 0, :]

        # gx_list
        _c = xp.expand_dims(gy, axis=1) * \
            xp.expand_dims(self.alpha, axis=2)  # just a product

        _d1 = xp.expand_dims(xp.expand_dims(- _a0, axis=1), axis=1)
        _d = _d1 * xp.expand_dims(self.alpha, axis=2) * \
            xp.expand_dims(q, axis=1)

        _e1 = _b0 * xp.expand_dims(self.alpha, axis=2)
        _e = _e1 * xp.expand_dims(q, axis=1)

        gx = _c + _d + _e

        gx_list = [gxi[:lens[i]] for (i, gxi) in enumerate(gx)]

        return tuple([gq, ] + gx_list)


def attention(q, xs):
    """Attension from Vector to Sequence by inner products.

    This function computes the attention and compute average
    feature vector from it.

    .. math::

      y_{i} = \\sum_{t} x_{t,i} \alpha_{t}
      \alpha_t = softmax(x_t^T q)

    """
    return Attention()(q, *xs)
