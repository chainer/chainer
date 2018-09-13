import numpy

import chainer
from chainer.backends import cuda


def get_array_module(*args):
    """Gets an appropriate one from :mod:`numpy` or :mod:`cupy`.

    This is almost equivalent to :func:`cupy.get_array_module`. The differences
    are that this function can be used even if CUDA is not available and that
    it will return their data arrays' array module for
    :class:`~chainer.Variable` arguments.

    Args:
        args: Values to determine whether NumPy or CuPy should be used.

    Returns:
        module: :mod:`cupy` or :mod:`numpy` is returned based on the types of
        the arguments.

    """
    if cuda.available:
        args = [arg.data if isinstance(arg, chainer.variable.Variable) else arg
                for arg in args]
        return cuda.cupy.get_array_module(*args)
    else:
        return numpy
