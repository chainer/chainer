import numpy
import six

from chainer import cuda
from chainer import function
from chainer.functions.math.matmul import _batch_matmul_gpu
from chainer.functions.math.matmul import _matmul
from chainer.utils import type_check


def _split(inputs, pos):
    return inputs[:pos], inputs[pos:]


def _seqs_to_array(xs, length, pad_value):
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


class AttentionScoreDot(function.Function):
    """Compute Attention Score by Inner Product"""

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
        batchsize = len(x_list)
        lens = list(map(len, x_list))
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

        return tuple([a[:lens[i]] for (i, a) in enumerate(alpha)])

    def forward_gpu(self, inputs):
        (q,), inputs = _split(inputs, 1)
        x_list = inputs
        batchsize = len(x_list)
        lens = list(map(len, x_list))
        max_len = max(lens)
        xmat = _seqs_to_array(x_list, max_len, 0)

        score = cuda.cupy.empty((batchsize, max_len, 1), dtype=xmat.dtype)
        _batch_matmul_gpu(xmat, q, score)
        score = score.reshape((batchsize, max_len))
        _mask_array(score, lens, -numpy.inf)

        # softmax
        alpha = score - score.max(axis=1, keepdims=True)
        cuda.cupy.exp(alpha, out=alpha)
        alpha /= alpha.sum(axis=1, keepdims=True)
        self.alpha = alpha

        return tuple([a[:lens[i]] for (i, a) in enumerate(alpha)])

    def backward_cpu(self, inputs, grads):
        q = inputs[0]
        x_list = inputs[1:]
        batchsize = len(x_list)
        lens = list(map(len, x_list))
        max_len = max(lens)
        xmat = _seqs_to_array(x_list, max_len, 0)
        gmat = _seqs_to_array(grads, max_len, 0)

        # gq
        _a0 = numpy.sum(gmat * self.alpha, axis=1)
        _a1 = numpy.sum(numpy.expand_dims(self.alpha, axis=2) * xmat, axis=1)
        _a = - numpy.expand_dims(_a0, axis=1) * _a1
        _b1 = numpy.sum(numpy.expand_dims(
            gmat * self.alpha, axis=2) * xmat, axis=1)
        gq = _a + _b1

        # gx_list
        gx_list = []
        for i in range(batchsize):
            gmati = gmat[i]
            alphai = self.alpha[i]
            qi = q[i]
            _c0 = - numpy.dot(gmati, alphai)
            _c = numpy.expand_dims(_c0 * alphai, axis=1) * \
                numpy.expand_dims(qi, axis=0)
            _d = numpy.expand_dims(gmati * alphai, axis=1) * \
                numpy.expand_dims(qi, axis=0)
            gx = _c + _d
            gx_list.append(gx[:lens[i]])

        return tuple([gq, ] + gx_list)

    def backward_gpu(self, inputs, grads):
        q = inputs[0]
        x_list = inputs[1:]
        lens = list(map(len, x_list))
        max_len = max(lens)
        xmat = _seqs_to_array(x_list, max_len, 0)
        gmat = _seqs_to_array(grads, max_len, 0)
        xp = cuda.cupy

        # gq
        _a0 = xp.sum(gmat * self.alpha, axis=1)
        _a1 = xp.sum(xp.expand_dims(self.alpha, axis=2) * xmat, axis=1)
        _a = - xp.expand_dims(_a0, axis=1) * _a1
        _b1 = xp.sum(xp.expand_dims(gmat * self.alpha, axis=2) * xmat, axis=1)
        gq = _a + _b1

        # gx_list
        _aq = xp.expand_dims(self.alpha, axis=2) * xp.expand_dims(q, axis=1)
        _c = - xp.expand_dims(xp.sum(self.alpha * gmat, axis=1), axis=1) + gmat
        gx = xp.expand_dims(_c, axis=2) * _aq
        gx_list = [gxi[:lens[i]] for (i, gxi) in enumerate(gx)]

        return tuple([gq, ] + gx_list)


def attention_score_dot(q, xs):
    return AttentionScoreDot()(q, *xs)
