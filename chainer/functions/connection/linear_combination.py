import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import cupy


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


class LinearCombination(function.Function):
    """Compute linear combinations for vactor and scalar sequences"""

    def check_type_forward(self, in_types):
        batchsize = len(in_types) / 2
        xs_types = in_types[:batchsize]
        cs_types = in_types[batchsize:]
        assert(len(xs_types) == batchsize)
        assert(len(cs_types) == batchsize)

        dim = xs_types[0].shape[1]
        for x_type, c_type in zip(xs_types, cs_types):
            type_check.expect(
                x_type.dtype == numpy.float32,
                c_type.dtype == numpy.float32,
                x_type.ndim == 2,
                c_type.ndim == 1,
                x_type.shape[1] == dim,
                x_type.shape[0] == c_type.shape[0],
            )

    def forward_cpu(self, inputs):
        batchsize = len(inputs) / 2
        x_list, c_list = _split(inputs, batchsize)
        lens = map(len, x_list)
        max_len = max(lens)
        xmat = _seqs_to_array(x_list, max_len, 0)
        cmat = _seqs_to_array(c_list, max_len, 0)

        out = numpy.sum(numpy.expand_dims(cmat, axis=2) * xmat, axis=1)
        self.out = out
        return out,

    def forward_gpu(self, inputs):
        batchsize = len(inputs) / 2
        x_list, c_list = _split(inputs, batchsize)
        lens = map(len, x_list)
        max_len = max(lens)
        xmat = _seqs_to_array(x_list, max_len, 0)
        cmat = _seqs_to_array(c_list, max_len, 0)

        out = cupy.sum(cupy.expand_dims(cmat, axis=2) * xmat, axis=1)
        self.out = out
        return out,

    def backward_cpu(self, inputs, grads):
        batchsize = len(inputs) / 2
        x_list, c_list = _split(inputs, batchsize)
        lens = map(len, x_list)
        max_len = max(lens)
        xmat = _seqs_to_array(x_list, max_len, 0)
        cmat = _seqs_to_array(c_list, max_len, 0)
        gy, = grads

        # gxs
        gxs = numpy.expand_dims(cmat, axis=2) * numpy.expand_dims(gy, axis=1)
        gx_list = [gx[:l, :] for (l, gx) in zip(lens, gxs)]

        # gcs
        gcs = numpy.sum(numpy.expand_dims(gy, axis=1) * xmat, axis=2)
        gc_list = [gc[:l] for (l, gc) in zip(lens, gcs)]

        return tuple(gx_list + gc_list)

    def backward_gpu(self, inputs, grads):
        batchsize = len(inputs) / 2
        x_list, c_list = _split(inputs, batchsize)
        lens = map(len, x_list)
        max_len = max(lens)
        xmat = _seqs_to_array(x_list, max_len, 0)
        cmat = _seqs_to_array(c_list, max_len, 0)
        gy, = grads

        # gxs
        gxs = cupy.expand_dims(cmat, axis=2) * cupy.expand_dims(gy, axis=1)
        gx_list = [gx[:l, :] for (l, gx) in zip(lens, gxs)]

        # gcs
        gcs = cupy.sum(cupy.expand_dims(gy, axis=1) * xmat, axis=2)
        gc_list = [gc[:l] for (l, gc) in zip(lens, gcs)]

        return tuple(gx_list + gc_list)


def linear_combination(xs, cs):
    return LinearCombination()(*(list(xs) + list(cs)))
