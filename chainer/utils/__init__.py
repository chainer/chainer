import numpy

from chainer.utils import walker_alias
from chainer.utils import future_warning as future_warning_

WalkerAlias = walker_alias.WalkerAlias
future_warning = future_warning_.future_warning


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
