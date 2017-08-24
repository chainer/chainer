import contextlib
import inspect

import numpy

from chainer.utils import walker_alias  # NOQA


# import class and function
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


def contextmanager(func):
    """A decorator used to define a factory function for ``with`` statement.

    This does exactlyl the same thing as ``@contextlib.contextmanager``, but
    with workaround for the issue that it does not transfer source file name
    and line number at which the original function was defined.

    .. seealso:: ``docs/source/conf.py``
    """

    sourcefile = inspect.getsourcefile(func)
    _, linenumber = inspect.getsourcelines(func)

    wrapper = contextlib.contextmanager(func)

    if sourcefile is not None:
        wrapper.__chainer_wrapped_sourcefile__ = sourcefile
        wrapper.__chainer_wrapped_linenumber__ = linenumber
    return wrapper
