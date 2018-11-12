import chainerx
from chainerx import _docs


def set_docs():
    _docs_creation()
    _docs_indexing()
    _docs_linalg()
    _docs_logic()
    _docs_manipulation()
    _docs_math()
    _docs_statistics()
    _docs_connection()
    _docs_normalization()
    _docs_pooling()


def _docs_creation():
    pass


def _docs_indexing():
    pass


def _docs_linalg():
    _docs.set_doc(
        chainerx.dot,
        """dot(a, b)
Returns a dot product of two arrays.

For arrays with more than one axis, it computes the dot product along the last
axis of ``a`` and the second-to-last axis of ``b``. This is just a matrix
product if the both arrays are 2-D. For 1-D arrays, it uses their unique axis
as an axis to take dot product over.

Args:
    a (~chainerx.ndarray): The left argument.
    b (~chainerx.ndarray): The right argument.

Returns:
    :class:`~chainerx.ndarray`: Output array.

Note:
    This function currently does not support N > 2 dimensional arrays.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to input arrays ``a`` and ``b``.

.. seealso:: :func:`numpy.dot`
""")


def _docs_logic():
    _docs.set_doc(
        chainerx.equal,
        """equal(x1, x2)
Returns an elementwise equality array.

Args:
    x1 (~chainerx.ndarray): Input array.
    x2 (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Output array of type bool.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :func:`numpy.equal`
""")


def _docs_manipulation():
    _docs.set_doc(
        chainerx.reshape,
        """reshape(a, newshape)
Returns a reshaped array.

Args:
    a (~chainerx.ndarray): Array to be reshaped.
    newshape (int or tuple of ints): The new shape of the array to return.
        If it is an integer, then it is treated as a tuple of length one.
        It should be compatible with ``a.size``. One of the elements can be
        -1, which is automatically replaced with the appropriate value to
        make the shape compatible with ``a.size``.

Returns:
    :class:`~chainerx.ndarray`: A reshaped view of ``a`` if possible,
    otherwise a copy.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``a``.

.. seealso:: :func:`numpy.reshape`
""")


def _docs_math():
    pass


def _docs_statistics():
    pass


def _docs_connection():
    pass


def _docs_normalization():
    pass


def _docs_pooling():
    pass
