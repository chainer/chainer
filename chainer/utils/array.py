import warnings

import numpy

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
