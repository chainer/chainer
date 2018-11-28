import numpy

import chainerx


# TODO(sonots): Implement in C++, especially in CUDA
def normal(loc=0.0, scale=1.0, size=None, device=None):
    """Draws random samples from a normal (Gaussian) distribution.

    This is currently equivalent to :func:`numpy.random.normal`
    wrapped by :func:`chainerx.array`, given the device argument.

    .. seealso:: :func:`numpy.random.normal`
    """
    a = numpy.random.normal(loc, scale, size)
    return chainerx.array(a, device=device, copy=False)


# TODO(sonots): Implement in C++, especially in CUDA
def uniform(low=0.0, high=1.0, size=None, device=None):
    """Draws samples from a uniform distribution.

    This is currently equivalent to :func:`numpy.random.normal`
    wrapped by :func:`chainerx.array`, given the device argument.

    .. seealso:: :func:`numpy.random.uniform`
    """
    a = numpy.random.uniform(low, high, size)
    return chainerx.array(a, device=device, copy=False)
