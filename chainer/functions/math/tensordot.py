import collections
import numpy
import six

from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer import utils
from chainer.utils import type_check

import inspect
import os


def _loc():
    frame = inspect.currentframe().f_back
    ret = '%s, %s, %s' % (os.path.basename(frame.f_code.co_filename),
                          frame.f_code.co_name, frame.f_lineno)
    return ret


def _tensordot(a, b, a_axes, b_axes, c_axes):
    
    xp = cuda.get_array_module(a)
    axes = (a_axes[1], b_axes[0])
    y = xp.tensordot(a, b, axes)

    trans = [None for i in range(y.ndim)]
    
    table_a = [1 if i in a_axes[0] else 0 for i in range(a.ndim)]
    table_a = numpy.cumsum(table_a) - 1
    # print('# {}, table_a:{}'.format(_loc(), table_a))
    for i, dst in enumerate(c_axes[0]):
        j = a_axes[0][i]
        j = table_a[j]
        trans[dst] = j

    table_b = [1 if i in b_axes[1] else 0 for i in range(b.ndim)]
    table_b = numpy.cumsum(table_b) - 1
    # print('# {}, table_b:{}'.format(_loc(), table_b))
    for i, dst in enumerate(c_axes[1]):
        j = b_axes[1][i]
        j = table_b[j]
        trans[dst] = j
        trans[dst] += len(a_axes[0])

    # print('# {}, trans:{}'.format(_loc(), trans))
        
    do_transpose = False
    for i in range(y.ndim):
        if trans[i] != i:
            do_transpose = True
    if do_transpose:
        # print('# {}, y.shape(befor):{}'.format(_loc(), y.shape))
        y = xp.transpose(y, trans)
        # print('# {}, y.shape(after):{}'.format(_loc(), y.shape))
    
    return y


class TensorDot(function_node.FunctionNode):

    def __init__(self, axes=2, a_axes=None, b_axes=None, c_axes=None):
        self.axes = axes
        self.a_axes = a_axes
        self.b_axes = b_axes
        self.c_axes = c_axes

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        a, b = inputs

        if self.a_axes is None or self.b_axes is None:
            a_axes = [None, None]
            b_axes = [None, None]
            axes = self.axes
            if isinstance(axes, collections.Sequence):
                if len(axes) != 2:
                    raise ValueError('Axes must consist of two arrays.')
                a_axes[1], b_axes[0] = axes
                if numpy.isscalar(a_axes[1]):
                    a_axes[1] = a_axes[1],
                if numpy.isscalar(b_axes[0]):
                    b_axes[0] = b_axes[0],
            else:
                a_axes[1] = list(six.moves.range(a.ndim - axes, a.ndim))
                b_axes[0] = list(six.moves.range(axes))
            a_axes[0] = [i for i in six.moves.range(a.ndim) if i not in a_axes[1]]
            b_axes[1] = [i for i in six.moves.range(b.ndim) if i not in b_axes[0]]
            self.a_axes = a_axes
            self.b_axes = b_axes

        if self.c_axes is None:
            c_axes = [None, None]
            c_axes[0] = list(six.moves.range(len(a_axes[0])))
            c_axes[1] = list(six.moves.range(len(a_axes[0]), len(a_axes[0]) + len(b_axes[1])))
            self.c_axes = c_axes
        
        # print('# {}, axes:{}'.format(_loc(), self.axes))
        # print('# {}. a.shape:{}'.format(_loc(), a.shape))
        # print('# {}, a_axes: {}'.format(_loc(), self.a_axes))
        # print('# {}. b.shape:{}'.format(_loc(), b.shape))
        # print('# {}, b_axes: {}'.format(_loc(), self.b_axes))
        # print('# {}, c_axes: {}'.format(_loc(), self.c_axes))

        y = _tensordot(a, b, self.a_axes, self.b_axes, self.c_axes)
        return utils.force_array(y),

    def backward(self, indexes, grad_outputs):
        a, b = self.get_retained_inputs()
        gy, = grad_outputs

        # print('# {}, compute ga'.format(_loc()))
        ga, = TensorDot(a_axes=self.c_axes,
                        b_axes=[self.b_axes[1], self.b_axes[0]],
                        c_axes=self.a_axes).apply((gy, b))
        # print('# {}, ga.shape:{}'.format(_loc(), ga.shape))

        # print('# {}, compute gb'.format(_loc()))
        gb, = TensorDot(a_axes=[self.a_axes[1], self.a_axes[0]],
                        b_axes=self.c_axes,
                        c_axes=self.b_axes).apply((a, gy))
        # print('# {}, gb.shape:{}'.format(_loc(), gb.shape))

        return ga, gb
        
def tensordot(a, b, axes=2):
    return TensorDot(axes=axes).apply((a, b))[0]
