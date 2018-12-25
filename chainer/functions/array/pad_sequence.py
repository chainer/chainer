import numpy
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class PadSequence(function_node.FunctionNode):

    """Padding arrays to create a matrix."""

    def __init__(self, length, padding):
        self.length = length
        self.padding = padding

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)

        for i, in_type in enumerate(in_types):
            type_check._argname((in_type,), ('x{}'.format(i),))
            type_check.expect(
                in_type.ndim > 0,
                in_type.shape[1:] == in_types[0].shape[1:],
                in_type.dtype == in_types[0].dtype
            )

        if self.length is not None:
            for in_type in in_types:
                type_check.expect(in_type.shape[0] <= self.length)

    def forward(self, xs):
        xp = backend.get_array_module(*xs)

        if self.length is None:
            length = max(len(x) for x in xs)
        else:
            length = self.length

        shape = (len(xs), length) + xs[0].shape[1:]
        y = xp.empty(shape, xs[0].dtype)
        if length == 0:
            return y,  # y is an empty array

        if xp is numpy or any(not x._c_contiguous for x in xs):
            for i, x in enumerate(xs):
                l = len(x)
                if l == length:
                    y[i] = x
                else:
                    y[i, 0:l] = x
                    y[i, l:] = self.padding
        else:
            # This code assumes that all arrays are c_contiguous
            ptr_shape = (Ellipsis,) + (None,) * xs[0].ndim
            ptrs = cuda.cupy.array(
                [x.data for x in xs], numpy.uintp)[ptr_shape]
            lengths = cuda.cupy.array(
                [len(x) for x in xs], numpy.int32)[ptr_shape]
            base = utils.size_of_shape(xs[0].shape[1:])
            cuda.elementwise(
                'P ptr, int32 length, T pad, int32 base, int32 max_length',
                'T y',
                '''
                int d = i / base % max_length;
                if (d < length) {
                  y = reinterpret_cast<const T*>(ptr)[i % (base * max_length)];
                } else {
                  y = pad;
                }
                ''',
                'pad_sequence_fwd'
            )(ptrs, lengths, self.padding, base, length, y)

        return y,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        inputs = self.inputs
        if gy.size == 0:
            # `split` in NumPy 1.9 behaves inconsistently when size is zero.
            gy = [gy]
        else:
            gy = chainer.functions.split_axis(gy, len(inputs), axis=0)
        return tuple(g[0, :x.shape[0]] for g, x in six.moves.zip(gy, inputs))


def pad_sequence(xs, length=None, padding=0):
    """Pad given arrays to make a matrix.

    Args:
        xs (list of ~chainer.Variable or :ref:`ndarray`):
            Variables you want to concatenate.
        length (None or int): Size of the first dimension of a padded array.
            If it is ``None``, the longest size of the first dimension of
            ``xs`` is used.
        padding (int or float): Value to fill.

    Returns:
        ~chainer.Variable: A padded matrix. Its shape is
        ``(n, length, ...)``, where ``n == len(xs)``.

    """
    return PadSequence(length, padding).apply((xs))[0]
