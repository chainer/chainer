import numpy

from chainer import cuda
from chainer import initializer


class Identity(initializer.Initializer):

    """Initializes array with the identity matrix.

    It initializes the given array with the constant
    multiple of the identity matrix.
    Note that arrays to be passed must be 2D squared matrices.

    Attributes:
        scale (scalar): A constant to be multiplied to identity matrices.

    """

    def __init__(self, scale=1.0, dtype=None):
        self.scale = scale
        super(Identity, self).__init__(dtype)

    def __call__(self, array):
        shape = array.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('Identity matrix initialization can only be used '
                             'for 2D squared matrices.')
        array[...] = 0
        xp = cuda.get_array_module(array)
        d = xp.diagonal(array)
        if xp == numpy:
            writeable = d.flags.writeable
            d.flags.writeable = True
        d[...] = self.scale
        if xp == numpy:
            d.flags.writeable = writeable


class Constant(initializer.Initializer):

    """Initializes array with constant value.

    Attributes:
        fill_value (scalar or numpy.ndarray or cupy.ndarray):
            A constant to be assigned to the initialized array.
            Broadcast is allowed on this assignment.
        dtype: Data type specifier.

    """

    def __init__(self, fill_value, dtype=None):
        self.fill_value = fill_value
        super(Constant, self).__init__(dtype)

    def __call__(self, array):
        xp = cuda.get_array_module(array)
        array[...] = xp.asarray(self.fill_value)


def Zero(dtype=None):
    """Returns initializer that initializes array with the all-zero array.

    Args:
        dtype: Data type specifier.

    Returns:
        numpy.ndarray or cupy.ndarray: An initialized array.

    """
    return Constant(0.0, dtype=dtype)


def One(dtype=None):
    """Returns initializer that initializes array with the all-one array.

    Args:
        dtype: Data type specifier.

    Returns:
        numpy.ndarray or cupy.ndarray: An initialized array.

    """
    return Constant(1.0, dtype=dtype)
