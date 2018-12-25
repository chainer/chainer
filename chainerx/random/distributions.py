import numpy

import chainerx


# TODO(sonots): Implement in C++, especially in CUDA
def normal(*args, **kwargs):
    """normal(*args, **kwargs, device=None)

    Draws random samples from a normal (Gaussian) distribution.

    This is currently equivalent to :func:`numpy.random.normal`
    wrapped by :func:`chainerx.array`, given the device argument.

    .. seealso:: :func:`numpy.random.normal`
    """
    device = kwargs.pop('device', None)

    a = numpy.random.normal(*args, **kwargs)
    return chainerx.array(a, device=device, copy=False)


# TODO(sonots): Implement in C++, especially in CUDA
def uniform(*args, **kwargs):
    """uniform(*args, **kwargs, device=None)

    Draws samples from a uniform distribution.

    This is currently equivalent to :func:`numpy.random.normal`
    wrapped by :func:`chainerx.array`, given the device argument.

    .. seealso:: :func:`numpy.random.uniform`
    """
    device = kwargs.pop('device', None)

    a = numpy.random.uniform(*args, **kwargs)
    return chainerx.array(a, device=device, copy=False)
