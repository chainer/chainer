import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


def _transpose(xs, length):
    if length == 0:
        return ()

    xp = backend.get_array_module(*xs)
    lengths = numpy.empty(length, dtype=numpy.int32)
    end = length
    for i, x in enumerate(xs):
        len_x = len(x)
        if len_x == end:
            continue
        lengths[len_x:end] = i
        end = len_x
    lengths[0:end] = len(xs)

    if xp is numpy:
        dtype = xs[0].dtype
        unit = xs[0].shape[1:]

        outs = tuple([xp.empty((l,) + unit, dtype=dtype) for l in lengths])
        for i, x in enumerate(xs):
            for p, xi in enumerate(x):
                outs[p][i] = xi

    else:
        offsets1 = numpy.empty(len(xs) + 1, dtype=numpy.int32)
        offsets1[0] = 0
        numpy.cumsum([len(x) for x in xs], out=offsets1[1:])

        offsets2 = numpy.empty(length + 1, dtype=numpy.int32)
        offsets2[0] = 0
        numpy.cumsum(lengths, dtype=numpy.int32, out=offsets2[1:])

        x = xp.concatenate(xs, axis=0)
        o = xp.empty_like(x)
        unit = xs[0].size // len(xs[0])
        size = length * len(xs) * unit
        cuda.elementwise(
            'int32 len, int32 unit, raw int32 off1, raw int32 off2, raw T vs',
            'raw T hs',
            '''
            int ind = i / unit;
            int off = i - ind * unit;
            int y = ind / len;
            int x = ind - y * len;
            if (off2[x] + y < off2[x + 1]) {
              hs[(off2[x] + y) * unit + off] = vs[(off1[y] + x) * unit + off];
            }
            ''',
            'transpose_sequence'
        )(length, unit, cuda.to_gpu(offsets1), cuda.to_gpu(offsets2), x, o,
          size=size)
        outs = tuple(xp.split(o, offsets2[1:-1]))

    return outs


class TransposeSequence(function_node.FunctionNode):

    """Function that transposes a list of Variables."""

    def __init__(self, length):
        self._length = length

    def check_type_forward(self, xs_type):
        for p, n in zip(xs_type, xs_type[1:]):
            type_check.expect(
                p.shape[0] >= n.shape[0],
                p.shape[1:] == n.shape[1:],
            )

    def forward(self, xs):
        if len(xs) == 0:
            return ()
        return _transpose(xs, self._length)

    def backward(self, indexes, grad_outputs):
        return TransposeSequence(len(self.inputs)).apply(grad_outputs)


def transpose_sequence(xs):
    """Transpose a list of Variables.

    This function transposes a list of :class:`~chainer.Variable`\\ s and
    returns a list of :class:`Variable`\\ s.
    For example a user gives ``[(0, 1, 2, 3), (4, 5), (6)]``, the function
    returns ``[(0, 4, 6), (1, 5), (2), (3)]``.
    Note that a given list needs to be sorted by each length of
    :class:`~chainer.Variable`.

    Args:
        xs (list of :class:`~chainer.Variable` or :ref:`ndarray`):
            Variables to transpose.

    Returns:
        tuple of :class:`~chainer.Variable`: Transposed list.

    .. admonition:: Example

        >>> lst = [chainer.Variable(np.array([1, 1, 1])),
        ...        chainer.Variable(np.array([2, 2])),
        ...        chainer.Variable(np.array([3]))]
        >>> lst
        [variable([1, 1, 1]), variable([2, 2]), variable([3])]
        >>> transposed = F.transpose_sequence(lst)
        >>> transposed
        (variable([1, 2, 3]), variable([1, 2]), variable([1]))

    """
    if len(xs) == 0:
        return ()
    return TransposeSequence(len(xs[0])).apply(xs)
