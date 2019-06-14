import numpy

import chainerx


# TODO(sonots): Support subclassing
def asanyarray(a, dtype=None, device=None):
    """Converts an object to an array.

    This is currently equivalent to :func:`~chainerx.asarray`, since there are
    no subclasses of ndarray in ChainerX. Note that the original
    :func:`numpy.asanyarray` returns the input array as is, if it is an
    instance of a subtype of :class:`numpy.ndarray`.

    .. seealso:: :func:`chainerx.asarray`, :func:`numpy.asanyarray`
    """
    return chainerx.asarray(a, dtype, device)


def fromfile(file, dtype=float, count=-1, sep='', device=None):
    """Constructs an array from data in a text or binary file.

    This is currently equivalent to :func:`numpy.fromfile`
    wrapped by :func:`chainerx.array`, given the device argument.

    .. seealso:: :func:`numpy.fromfile`

    """
    return chainerx.array(
        numpy.fromfile(
            file, dtype=dtype, count=count, sep=sep),
        device=device)


def fromfunction(function, shape, **kwargs):
    """ Constructs an array by executing a function over each coordinate.

    This is currently equivalent to :func:`numpy.fromfunction`
    wrapped by :func:`chainerx.array`, given the device argument.

    Note:
        Keywords other than ``dtype`` and ``device`` are passed to
        ```function```.

    .. seealso:: :func:`numpy.fromfunction`

    """
    dtype = kwargs.pop('dtype', float)
    device = kwargs.pop('device', None)
    return chainerx.array(
        numpy.fromfunction(
            function, shape, dtype=dtype, **kwargs),
        device=device)


# TODO(hvy): Optimize with pre-allocated memory using count for non-native
# devices.
def fromiter(iterable, dtype, count=-1, device=None):
    """Constructs a new 1-D array from an iterable object.

    This is currently equivalent to :func:`numpy.fromiter`
    wrapped by :func:`chainerx.array`, given the device argument.

    .. seealso:: :func:`numpy.fromiter`

    """
    return chainerx.array(
        numpy.fromiter(iterable, dtype=dtype, count=count),
        device=device)


def fromstring(string, dtype=float, count=-1, sep='', device=None):
    """Constructs a new 1-D array initialized from text data in a string.

    This is currently equivalent to :func:`numpy.fromstring`
    wrapped by :func:`chainerx.array`, given the device argument.

    .. seealso:: :func:`numpy.fromstring`

    """
    # sep should always be specified in numpy.fromstring since its default
    # argument has been deprecated since 1.14.
    # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.fromstring.html
    return chainerx.array(
        numpy.fromstring(
            string, dtype=dtype, count=count, sep=sep),
        device=device)


def loadtxt(
        fname, dtype=float, comments='#', delimiter=None, converters=None,
        skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes',
        device=None):
    """Constructs an array by loading data from a text file.

    This is currently equivalent to :func:`numpy.loadtxt`
    wrapped by :func:`chainerx.array`, given the device argument.

    .. seealso:: :func:`numpy.loadtxt`

    """
    return chainerx.array(
        numpy.loadtxt(
            fname, dtype=dtype, comments=comments, delimiter=delimiter,
            converters=converters, skiprows=skiprows, usecols=usecols,
            unpack=unpack, ndmin=ndmin, encoding=encoding),
        device=device)
