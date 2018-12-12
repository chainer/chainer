import warnings

import numpy
import six

import chainer
from chainer.backends import cuda


def as_vec(x):
    warnings.warn(
        'chainer.utils.array.as_vec is deprecated. Please refer to '
        'numpy.ravel or other array backend functions to flatten ndarrays.',
        DeprecationWarning)
    if x.ndim == 1:
        return x
    return x.ravel()


def as_mat(x):
    warnings.warn(
        'chainer.utils.array.as_mat is deprecated. Please refer to '
        'numpy.reshape or other array backend functions to reshape ndarrays.',
        DeprecationWarning)
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


def empty_like(x):
    warnings.warn(
        'chainer.utils.array.empty_like is deprecated. Please refer to '
        'numpy.empty_like or other array backend functions to initialize '
        'empty arrays.',
        DeprecationWarning)
    if cuda.available and isinstance(x, cuda.ndarray):
        return cuda.cupy.empty_like(x)
    else:
        return numpy.empty_like(x)


def size_of_shape(shape):
    size = 1
    for i in shape:
        size *= i

    # should not return long in Python 2
    return int(size)


def sum_to(x, shape):
    if x.shape == shape:
        return x
    if isinstance(x, chainer.Variable):
        raise TypeError(
            'chainer.utils.sum_to does not support Variable input. '
            'Use chainer.functions.sum_to instead.')
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(six.moves.range(lead))
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y
