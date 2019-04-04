import collections
import contextlib
import shutil
import sys
import tempfile

import numpy
import six

import chainer
# import classes and functions
from chainer.utils.array import size_of_shape  # NOQA
from chainer.utils.array import sum_to  # NOQA
from chainer.utils.conv import get_conv_outsize  # NOQA
from chainer.utils.conv import get_deconv_outsize  # NOQA
from chainer.utils.error import _format_array_props  # NOQA
from chainer.utils.experimental import experimental  # NOQA
from chainer.utils.meta import enable_final  # NOQA
from chainer.utils.meta import final  # NOQA
from chainer.utils.nondeterministic import nondeterministic  # NOQA
from chainer.utils.sparse import CooMatrix  # NOQA
from chainer.utils.sparse import get_order  # NOQA
from chainer.utils.sparse import to_coo  # NOQA

# The following alias has been moved to chainer/__init__.py in order to break
# circular imports in Python 2.
# from chainer.utils.walker_alias import WalkerAlias


# TODO(kmaehashi) remove this when `six.moves.collections_abc` is implemented.
# See: https://github.com/chainer/chainer/issues/5097
try:
    collections_abc = collections.abc  # type: ignore
except AttributeError:  # python <3.3
    collections_abc = collections  # type: ignore


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


def _repr_with_named_data(inst, **kwargs):
    """Convenient function to generate `repr` string with custom named data"""
    if six.PY2:
        class_name = inst.__class__.__name__
    else:
        class_name = inst.__class__.__qualname__
    return '<{}.{} {}>'.format(
        inst.__module__, class_name,
        ' '.join('{}={}'.format(k, v) for k, v in six.iteritems(kwargs)))


def _check_arrays_forward_compatible(arrays, label=None):
    if not chainer.is_arrays_compatible(arrays):
        raise TypeError(
            'incompatible array types are mixed in the forward input{}.\n'
            'Actual: {}'.format(
                ' ({})'.format(label) if label is not None else '',
                ', '.join(str(type(a)) for a in arrays)))


def _raise_from(exc_type, message, orig_exc):
    # Raises an exception that wraps another exception.
    message = (
        '{}\n\n'
        '(caused by)\n'
        '{}: {}\n'.format(message, type(orig_exc).__name__, orig_exc))
    new_exc = exc_type(message)
    if sys.version_info < (3,):
        six.reraise(exc_type, new_exc, sys.exc_info()[2])
    else:
        six.raise_from(new_exc.with_traceback(orig_exc.__traceback__), None)
