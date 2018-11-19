import numpy

import chainerx


# TODO(sonots): Implement in C++
def normal(*args, device=None, **kwargs):
    """Draws random samples from a normal (Gaussian) distribution.

    This is currently equivalent to :func:`numpy.random.normal`
    wrapped by :func:`chainerx.array`, given the device argument.

    .. seealso:: :func:`numpy.random.normal`
    """
    a = numpy.random.normal(*args, **kwargs)
    return chainerx.array(a, device=device, copy=False)


# TODO(sonots): Implement in C++
def uniform(*args, device=None, **kwargs):
    """Draws samples from a uniform distribution.

    This is currently equivalent to :func:`numpy.random.normal`
    wrapped by :func:`chainerx.array`, given the device argument.

    .. seealso:: :func:`numpy.random.uniform`
    """
    a = numpy.random.uniform(*args, **kwargs)
    return chainerx.array(a, device=device, copy=False)
