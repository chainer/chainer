import chainerx
from chainerx import _docs


def set_docs():
    ndarray = chainerx.ndarray

    _docs.set_doc(
        ndarray,
        """ndarray(shape, dtype, device=None)
Multi-dimensional array.

Args:
    shape (tuple of ints): Shape of created array.
    dtype: Data type.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

.. seealso:: :class:`numpy.ndarray`
""")
