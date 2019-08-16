import chainerx
import numpy


# TODO(ecastill): Implement in C++
def broadcast_arrays(*args):
    """Broadcast any number of arrays against each other.

    It tries to return a view if possible, otherwise returns a copy.

    Args:
        *args (~chainerx.ndarray): The arrays to broadcast.

    Returns:
        [chainerx.ndarray]: list of arrays
        These arrays are views on the original arrays.  They are typically
        not contiguous.  Furthermore, more than one element of a
        broadcasted array may refer to a single memory location.  If you
        need to write to the arrays, make copies first.
    """
    common_shape = numpy.broadcast(*args).shape
    return [chainerx.broadcast_to(a, common_shape) for a in args]
