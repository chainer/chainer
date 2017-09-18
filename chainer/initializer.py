import numpy


class Initializer(object):
    """Initializes array.

    It initializes the given array.

    Attributes:
        ~Initializer.dtype: Data type specifier. It is for type check in
            ``__call__`` function.

    """

    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, array):
        """Initializes given array.

        This method destructively changes the value of array.
        The derived class is required to implement this method.
        The algorithms used to make the new values depend on the
        concrete derived classes.

        Args:
            array (numpy.ndarray or cupy.ndarray):
                An array to be initialized by this initializer.

        """
        raise NotImplementedError()


# Original code forked from MIT licensed keras project
# https://github.com/fchollet/keras/blob/master/keras/initializations.py

def get_fans(shape):
    if not isinstance(shape, tuple):
        raise ValueError('shape must be tuple')

    if len(shape) < 2:
        raise ValueError('shape must be of length >= 2: shape={}', shape)

    receptive_field_size = numpy.prod(shape[2:], dtype=numpy.int32)
    fan_in = shape[1] * receptive_field_size
    fan_out = shape[0] * receptive_field_size
    return fan_in, fan_out
