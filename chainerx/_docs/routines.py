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
        chainerx.logical_not,
        """logical_not(x)
Returns an array of NOT x element-wise.

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Output array of type bool.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :func:`numpy.logical_not`
""")

    _docs.set_doc(
        chainerx.greater,
        """greater(x1, x2)
Returns an array of (x1 > x2) element-wise.

Args:
    x1 (~chainerx.ndarray): Input array.
    x2 (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Output array of type bool.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :func:`numpy.greater`
""")

    _docs.set_doc(
        chainerx.greater_equal,
        """greater_equal(x1, x2)
Returns an array of (x1 >= x2) element-wise.

Args:
    x1 (~chainerx.ndarray): Input array.
    x2 (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Output array of type bool.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :func:`numpy.greater_equal`
""")

    _docs.set_doc(
        chainerx.less,
        """less(x1, x2)
Returns an array of (x1 < x2) element-wise.

Args:
    x1 (~chainerx.ndarray): Input array.
    x2 (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Output array of type bool.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :func:`numpy.less`
""")

    _docs.set_doc(
        chainerx.less_equal,
        """less_equal(x1, x2)
Returns an array of (x1 <= x2) element-wise.

Args:
    x1 (~chainerx.ndarray): Input array.
    x2 (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Output array of type bool.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :func:`numpy.less_equal`
""")

    _docs.set_doc(
        chainerx.equal,
        """equal(x1, x2)
Returns an array of (x1 == x2) element-wise.

Args:
    x1 (~chainerx.ndarray): Input array.
    x2 (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Output array of type bool.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :func:`numpy.equal`
""")

    _docs.set_doc(
        chainerx.not_equal,
        """not_equal(x1, x2)
Returns an array of (x1 != x2) element-wise.

Args:
    x1 (~chainerx.ndarray): Input array.
    x2 (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Output array of type bool.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :func:`numpy.not_equal`
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
