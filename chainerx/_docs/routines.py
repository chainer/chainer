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
    _docs.set_doc(
        chainerx.empty,
        """empty(shape, dtype, device=None)
Creates an uninitialized array with given shape and dtype.

Args:
    shape (tuple of ints): Shape of the array.
    dtype: Data type of the array.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    :class:`~chainerx.ndarray`: New uninitialized array.

.. seealso:: :func:`numpy.empty`
""")

    _docs.set_doc(
        chainerx.empty_like,
        """empty_like(a, device=None)
Creates an uninitialized array with the same shape and dtype as a given array.

Args:
    a (~chainerx.ndarray): Prototype array.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    :class:`~chainerx.ndarray`: New uninitialized array.

Warning:
    If ``device`` argument is omitted, the new array is created on the default
    device, not the device of the prototype array.

.. seealso:: :func:`numpy.empty_like`
""")

    _docs.set_doc(
        chainerx.eye,
        """eye(N, M=None, k=0, dtype=float64, device=None)
Returns a 2-D array with ones on the diagonals and zeros elsewhere.

Args:
    N (int): Number of rows.
    M (int): Number of columns. M == N by default.
    k (int): Index of the diagonal. Zero indicates the main diagonal,
        a positive index an upper diagonal, and a negative index a lower
        diagonal.
    dtype: Data type.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: A 2-D array with given diagonals filled with ones and
    zeros elsewhere.

.. seealso:: :func:`numpy.eye`
""")

    _docs.set_doc(
        chainerx.identity,
        """identity(n, dtype=None, device=None)
Returns a 2-D identity array.

It is equivalent to ``eye(n, n, dtype)``.

Args:
    n (int): Number of rows and columns.
    dtype: Data type.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: A 2-D identity array.

.. seealso:: :func:`numpy.identity`
""")

    _docs.set_doc(
        chainerx.ones,
        """ones(shape, dtype, device=None)
Returns a new array of given shape and dtype, filled with ones.

Args:
    shape (tuple of ints): Shape of the array.
    dtype: Data type.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: New array.

.. seealso:: :func:`numpy.ones`
""")

    _docs.set_doc(
        chainerx.ones_like,
        """ones_like(a, device=None)
Returns an array of ones with same shape and dtype as a given array.

Args:
    a (~chainerx.ndarray): Prototype array.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: New array.

Warning:
    If ``device`` argument is omitted, the new array is created on the default
    device, not the device of the prototype array.

.. seealso:: :func:`numpy.ones_like`
""")

    _docs.set_doc(
        chainerx.zeros,
        """zeros(shape, dtype, device=None)
Returns a new array of given shape and dtype, filled with zeros.

Args:
    shape (tuple of ints): Shape of the array.
    dtype: Data type.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: New array.

.. seealso:: :func:`numpy.zeros`
""")

    _docs.set_doc(
        chainerx.zeros_like,
        """zeros_like(a, device=None)
Returns an array of zeros with same shape and dtype as a given array.

Args:
    a (~chainerx.ndarray): Prototype array.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: New array.

Warning:
    If ``device`` argument is omitted, the new array is created on the default
    device, not the device of the prototype array.

.. seealso:: :func:`numpy.zeros_like`
""")

    _docs.set_doc(
        chainerx.full,
        """full(shape, fill_value, dtype, device=None)
Returns a new array of given shape and dtype, filled with a given value.

Args:
    shape (tuple of ints): Shape of the array.
    dtype: Data type.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: New array.

.. seealso:: :func:`numpy.full`
""")

    _docs.set_doc(
        chainerx.full_like,
        """full_like(a, fill_value, dtype=None, device=None)
Returns a full array with same shape and dtype as a given array.

Args:
    a (~chainerx.ndarray): Prototype array.
    dtype: Data type.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: New array.

Warning:
    If ``device`` argument is omitted, the new array is created on the default
    device, not the device of the prototype array.

.. seealso:: :func:`numpy.full_like`
""")

    _docs.set_doc(
        chainerx.array,
        """array(object, dtype=None, copy=True, device=None)
Creates an array.

Args:
    object: A :class:`chainerx.ndarray` object or any other object that can be
        passed to :func:`numpy.array`.
    dtype: Data type. If omitted, it's inferred from the input.
    copy (bool): If `False`, this function returns ``object`` if possible.
        Otherwise this function always returns a new array.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: New array.

Warning:
    If ``device`` argument is omitted, the new array is created on the default
    device, not the device of the input array.

.. seealso:: :func:`numpy.array`
""")

    _docs.set_doc(
        chainerx.asarray,
        """asarray(a, dtype=None, device=None)
Converts an object to an array.

Args:
    a: The source object.
    dtype: Data type. If omitted, it's inferred from the input.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: Array interpretation of ``a``.

Warning:
    If ``device`` argument is omitted, the new array is created on the default
    device, not the device of the input array.

.. seealso:: :func:`numpy.asarray`
""")

    _docs.set_doc(
        chainerx.ascontiguousarray,
        """ascontiguousarray(a, dtype=None, device=None)
Returns a C-contiguous array.

Args:
    a (~chainerx.ndarray): Source array.
    dtype: Data type.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: Contiguous array.

Warning:
    If ``device`` argument is omitted, the new array is created on the default
    device, not the device of the input array.

.. seealso:: :func:`numpy.ascontiguousarray`
""")

    _docs.set_doc(
        chainerx.copy,
        """copy(a)
Creates a copy of a given array.

Args:
    a (~chainerx.ndarray): Source array.

Returns:
    ~chainerx.ndarray: A copy array.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``a``.

.. seealso:: :func:`numpy.copy`
""")


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
    pass


def _docs_connection():
    pass


def _docs_normalization():
    pass


def _docs_pooling():
    pass
