import contextlib
import shutil
import tempfile

import numpy

from chainer.utils import walker_alias  # NOQA


# import class and function
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


@contextlib.contextmanager
def tempdir(**kwargs):
    # A context manager that defines a lifetime of a temporary directory.
    ignore_errors = kwargs.pop('ignore_errors', False)

    temp_dir = tempfile.mkdtemp(**kwargs)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=ignore_errors)
