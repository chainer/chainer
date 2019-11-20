import chainerx
from chainerx import _docs


def set_docs():
    _docs.set_doc(
        chainerx.to_numpy,
        """to_numpy(array, copy=True)
Converts a ChainerX array to NumPy

Args:
    array (~chainerx.ndarray): ChainerX array.
    copy (bool): If ``True``, a copy is always made. Otherwise, the resulting
        array may be aliased with the input array.

Returns:
    numpy.ndarray: NumPy array.
""")
