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
Returns an array without initializing the elements.

Args:
    shape (tuple of ints): Shape of the array.
    dtype: Data type of the array.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    :class:`~chainerx.ndarray`: New array with elements not initialized.

.. seealso:: :func:`numpy.empty`
""")

    _docs.set_doc(
        chainerx.empty_like,
        """empty_like(a, device=None)
Returns a new array with same shape and dtype of a given array.

Args:
    a (~chainerx.ndarray): Prototype array.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    :class:`~chainerx.ndarray`: New array with same shape and dtype as ``a`` \
with elements not initialized.

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
    object: A :class:`~chainerx.ndarray` object or any other object that can be
        passed to :func:`numpy.array`.
    dtype: Data type. If omitted, it's inferred from the input.
    copy (bool): If ``True``, the object is always copied. Otherwise, a copy
        will only be made if it is needed to satisfy any of the other
        requirements (dtype, device, etc.).
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
    ~chainerx.ndarray: Array interpretation of ``a``. If ``a`` is already an \
ndarray on the given device with matching dtype, no copy is performed.

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
    ~chainerx.ndarray: C-contiguous array. A copy will be made only if needed.

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
    ~chainerx.ndarray: A copy array on the same device as ``a``.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``a``.

.. seealso:: :func:`numpy.copy`
""")


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
        chainerx.logical_not,
        """logical_not(x)
Returns an array of NOT x element-wise.

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Output array of type bool.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :data:`numpy.logical_not`
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

.. seealso:: :data:`numpy.greater`
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

.. seealso:: :data:`numpy.greater_equal`
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

.. seealso:: :data:`numpy.less`
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

.. seealso:: :data:`numpy.less_equal`
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

.. seealso:: :data:`numpy.equal`
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

.. seealso:: :data:`numpy.not_equal`
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
    _docs.set_doc(
        chainerx.negative,
        """negative(x)
Numerical negative, element-wise.

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = -x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.negative`
""")

    _docs.set_doc(
        chainerx.add,
        """add(x1, x2)
Add arguments, element-wise.

Args:
    x1 (~chainerx.ndarray or scalar): Input array.
    x2 (~chainerx.ndarray or scalar): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = x_1 + x_2`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input arrays ``x1`` and ``x2``.

.. seealso:: :data:`numpy.add`
""")

    _docs.set_doc(
        chainerx.subtract,
        """subtract(x1, x2)
Subtract arguments, element-wise.

Args:
    x1 (~chainerx.ndarray or scalar): Input array.
    x2 (~chainerx.ndarray or scalar): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = x_1 - x_2`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input arrays ``x1`` and ``x2``.

.. seealso:: :data:`numpy.subtract`
""")

    _docs.set_doc(
        chainerx.multiply,
        """multiply(x1, x2)
Multiply arguments, element-wise.

Args:
    x1 (~chainerx.ndarray or scalar): Input array.
    x2 (~chainerx.ndarray or scalar): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = x_1 \\times x_2`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input arrays ``x1`` and ``x2``.

.. seealso:: :data:`numpy.multiply`
""")

    _docs.set_doc(
        chainerx.divide,
        """divide(x1, x2)
Divide arguments, element-wise.

Args:
    x1 (~chainerx.ndarray or scalar): Input array.
    x2 (~chainerx.ndarray or scalar): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\frac{x_1}{x_2}`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input arrays ``x1`` and ``x2``.

.. seealso:: :data:`numpy.divide`
""")

    _docs.set_doc(
        chainerx.sum,
        """sum(a, axis=None, keepdims=False)
Sum of array elements over a given axis.

Args:
    a (~chainerx.ndarray): Input array.
    axis (None or int or tuple of ints):
        Axis or axes along which a sum is performed.
        The flattened array is used by default.
    keepdims (bool):
        If this is set to ``True``, the reduced axes are left in the result
        as dimensions with size one.

Returns:
    :class:`~chainerx.ndarray`: The sum of input elements over a given axis.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``a``.

.. seealso:: :func:`numpy.sum`
""")

    _docs.set_doc(
        chainerx.maximum,
        """maximum(x1, x2)
Maximum arguments, element-wise.

Args:
    x1 (~chainerx.ndarray or scalar): Input array.
    x2 (~chainerx.ndarray or scalar): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = max(\{x_1, x_2\})`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input arrays ``x1`` and ``x2``.

Note:
    maximum of :class:`~chainerx.ndarray` and :class:`~chainerx.ndarray` is
    not supported yet.

.. seealso:: :data:`numpy.maximum`
""")

    _docs.set_doc(
        chainerx.exp,
        """exp(x)
Numerical exponential, element-wise.

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\exp x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.exp`
""")

    _docs.set_doc(
        chainerx.log,
        """log(x)
Natural logarithm, element-wise.

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\ln x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.log`
""")

    _docs.set_doc(
        chainerx.logsumexp,
        """logsumexp(x, axis=None, keepdims=False)
The log of the sum of exponentials of input array.

Args:
    x (~chainerx.ndarray): Input array.
    axis (None or int or tuple of ints):
        Axis or axes along which a sum is performed.
        The flattened array is used by default.
    keepdims (bool):
        If this is set to ``True``, the reduced axes are left in the result
        as dimensions with size one.

Returns:
    :class:`~chainerx.ndarray`: The log of the sum of exponentials of
    input elements over a given axis.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.
""")

    _docs.set_doc(
        chainerx.log_softmax,
        """log_softmax(x, axis=None)
The log of the softmax of input array.

Args:
    x (~chainerx.ndarray): Input array.
    axis (None or int or tuple of ints):
        Axis or axes along which a sum is performed.
        The flattened array is used by default.

Returns:
    :class:`~chainerx.ndarray`: The log of the softmax of input elements
    over a given axis.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.
""")

    _docs.set_doc(
        chainerx.sqrt,
        """sqrt(x)
Non-negative square-root, element-wise

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\sqrt x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.sqrt`
""")

    _docs.set_doc(
        chainerx.tanh,
        """tanh(x)
Hyperbolic tangent, element-wise

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\tanh x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.tanh`
""")

    _docs.set_doc(
        chainerx.isnan,
        """isnan(x)
Test element-wise for NaN and return result as a boolean array.

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: True where ``x`` is NaN, false otherwise

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :data:`numpy.isnan`
""")

    _docs.set_doc(
        chainerx.isinf,
        """isinf(x)
Test element-wise for positive or negative infinity.

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: True where ``x`` is positive or negative
    infinity, false otherwise.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :data:`numpy.isinf`
""")


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
