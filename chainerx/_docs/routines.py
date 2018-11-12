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
    pass


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
    _docs.set_doc(
        chainerx.amax,
        """amax(a, axis=None, keepdims=False)
Returns the maximum of an array or the maximum along an axis.

Note:
    When at least one element is NaN, the corresponding max value will be NaN.

Args:
    a (~chainerx.ndarray): Array to take the maximum.
    axis (None or int or tuple of ints): Along which axis to take the maximum.
        The flattened array is used by default.
        If this is a tuple of ints, the maximum is selected over multiple
        axes, instead of a single axis or all the axes.
    keepdims (bool): If ``True``, the axis is remained as an axis of size one.

Returns:
    :class:`~chainerx.ndarray`: The maximum of ``a``, along the axis if
    specified.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``a``.

.. seealso:: :func:`numpy.amax`
""")


def _docs_connection():
    pass


def _docs_normalization():
    pass


def _docs_pooling():
    pass
