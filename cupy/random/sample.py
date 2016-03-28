from cupy.random import distributions
from cupy.random import generator


def rand(*size, **kwarg):
    """Returns an array of uniform random values over the interval ``[0, 1)``.

    Each element of the array is uniformly distributed on the half-open
    interval ``[0, 1)``. All elements are identically and independently
    distributed (i.i.d.).

    Args:
        size (tuple of ints): The shape of the array.
        dtype: Data type specifier. Only float32 and float64 types are allowed.
            The default is float64.

    Returns:
        cupy.ndarray: A random array.

    .. seealso:: :func:`numpy.random.rand`

    """
    dtype = kwarg.pop('dtype', float)
    if kwarg:
        raise TypeError('rand() got unexpected keyword arguments %s'
                        % ', '.join(kwarg.keys()))
    return random_sample(size=size, dtype=dtype)


def randn(*size, **kwarg):
    """Returns an array of standard normal random values.

    Each element of the array is normally distributed with zero mean and unit
    variance. All elements are identically and independently distributed
    (i.i.d.).

    Args:
        size (tuple of ints): The shape of the array.
        dtype: Data type specifier. Only float32 and float64 types are allowed.
            The default is float64.

    Returns:
        cupy.ndarray: An array of standard normal random values.

    .. seealso:: :func:`numpy.random.randn`

    """
    dtype = kwarg.pop('dtype', float)
    if kwarg:
        raise TypeError('randn() got unexpected keyword arguments %s'
                        % ', '.join(kwarg.keys()))
    return distributions.normal(size=size, dtype=dtype)


def randint(low, high=None, size=None):
    """Returns a scalar or an array of integer values over ``[low, high)``.

    Each element of returned values are independently sampled from
    uniform distribution over left-close and right-open interval
    ``[low, high)``.

    Args:
        low (int): If ``high`` is not ``None``,
            it is the lower bound of the interval.
            Otherwise, it is the **upper** bound of the interval
            and lower bound of the inteval is set to ``0``.
        high (int): Upper bound of the interval.
        size (None or int or tuple of ints): The shape of returned value.

    Returns:
        int or cupy.ndarray of ints: If size is ``None``,
        it is single integer sampled.
        If size is integer, it is the 1D-array of length ``size`` element.
        Otherwise, it is the array whose shape specified by ``size``.
    """
    if high is None:
        lo = 0
        hi = low
    else:
        lo = low
        hi = high

    if lo >= hi:
        raise ValueError('low >= high')

    diff = hi - lo - 1
    rs = generator.get_random_state()
    return lo + rs.interval(diff, size)


def random_integers(low, high=None, size=None):
    """Return a scalar or an array of interger values over ``[low, high]``

    Each element of returned values are independently sampled from
    uniform distribution over closed interval ``[low, high]``.

    Args:
        low (int): If ``high`` is not ``None``,
            it is the lower bound of the interval.
            Otherwise, it is the **upper** bound of the interval
            and the lower bound is set to ``1``.
        high (int): Upper bound of the interval.
        size (None or int or tuple of ints): The shape of returned value.

    Returns:
        int or cupy.ndarray of ints: If size is ``None``,
        it is single integer sampled.
        If size is integer, it is the 1D-array of length ``size`` element.
        Otherwise, it is the array whose shape specified by ``size``.
    """
    if high is None:
        high = low
        low = 1
    return randint(low, high + 1, size)


def random_sample(size=None, dtype=float):
    """Returns an array of random values over the interval ``[0, 1)``.

    This is a variant of :func:`cupy.random.rand`.

    Args:
        size (int or tuple of ints): The shape of the array.
        dtype: Data type specifier. Only float32 and float64 types are allowed.

    Returns:
        cupy.ndarray: An array of uniformly distributed random values.

    .. seealso:: :func:`numpy.random.random_sample`

    """
    rs = generator.get_random_state()
    return rs.random_sample(size=size, dtype=dtype)


# TODO(okuta): Implement choice
