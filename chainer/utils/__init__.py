import numpy

from chainer.utils import argument  # NOQA
from chainer.utils import array  # NOQA
from chainer.utils import conv_nd  # NOQA
from chainer.utils import conv_nd_kernel  # NOQA
from chainer.utils import imgproc  # NOQA
from chainer.utils import type_check  # NOQA

# import classes and functions
from chainer.utils.conv import get_conv_outsize  # NOQA
from chainer.utils.conv import get_deconv_outsize  # NOQA
from chainer.utils.experimental import experimental  # NOQA
from chainer.utils.walker_alias import WalkerAlias  # NOQA


def force_array(x, dtype=None):
    # numpy returns a float value (scalar) when a return value of an operator
    # is a 0-dimension array.
    # We need to convert such a value to a 0-dimension array because `Function`
    # object needs to return an `numpy.ndarray`.
    if numpy.isscalar(x):
        if dtype is None:
            return numpy.array(x)
        else:
            return numpy.array(x, dtype)
    else:
        if dtype is None:
            return x
        else:
            return x.astype(dtype, copy=False)


def force_type(dtype, value):
    if numpy.isscalar(value):
        return dtype.type(value)
    elif value.dtype != dtype:
        return value.astype(dtype, copy=False)
    else:
        return value
