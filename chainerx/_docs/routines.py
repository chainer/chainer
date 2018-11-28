import chainerx
from chainerx import _docs


def set_docs():
    _docs_creation()
    _docs_indexing()
    _docs_linalg()
    _docs_logic()
    _docs_manipulation()
    _docs_math()
    _docs_sorting()
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

    _docs.set_doc(
        chainerx.frombuffer,
        """frombuffer(buffer, dtype=float, count=-1, offset=0, device=None)
Returns a 1-D array interpretation of a buffer.

The given ``buffer`` memory must be usable on the given device, otherwise,
an error is raised.

Note:
    The ``native`` backend requires a buffer of main memory, and
    the ``cuda`` backend requires a buffer of CUDA memory.
    No copy is performed.

Args:
    buffer: An object that exposes the buffer interface.
    dtype: Data type of the returned array.
    count (int): Number of items to read. -1 means all data in the buffer.
    offset (int): Start reading the buffer from this offset (in bytes).
    device (~chainerx.Device): Device of the returned array.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: 1-D array interpretation of ``buffer``.

.. seealso:: :func:`numpy.frombuffer`
""")

    _docs.set_doc(
        chainerx.arange,
        """arange([start=0, ]stop, [step=1, ]dtype=None, device=None)
Returns an array with  evenly spaced values within a given interval.

Values are generated within the half-open interval [``start``, ``stop``).
The first three arguments are mapped like the ``range`` built-in function,
i.e. ``start`` and ``step`` are optional.

Args:
    start: Start of the interval.
    stop: End of the interval.
    step: Step width between each pair of consecutive values.
    dtype: Data type specifier. It is inferred from other arguments by
        default.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: The 1-D array of range values.

.. seealso:: :func:`numpy.arange`
""")

    _docs.set_doc(
        chainerx.linspace,
        """linspace(start, stop, num=50, endpoint=True, dtype=None, device=None)
Returns an array with evenly spaced numbers over a specified interval.

Instead of specifying the step width like :func:`chainerx.arange()`,
this function requires the total number of elements specified.

Args:
    start: Start of the interval.
    stop: End of the interval.
    num: Number of elements.
    endpoint (bool): If ``True``, the stop value is included as the last
        element. Otherwise, the stop value is omitted.
    dtype: Data type specifier. It is inferred from the start and stop
        arguments by default.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: The 1-D array of ranged values.

.. seealso:: :func:`numpy.linspace`
""")  # NOQA

    _docs.set_doc(
        chainerx.diag,
        """diag(v, k=0, device=None)
Returns a diagonal or a diagonal array.

Args:
    v (~chainerx.ndarray): Array object.
    k (int): Index of diagonals. Zero indicates the main diagonal, a
        positive value an upper diagonal, and a negative value a lower
        diagonal.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: If ``v`` is a 1-D array, then it returns a 2-D
    array with the specified diagonal filled by ``v``. If ``v`` is a
    2-D array, then it returns the specified diagonal of ``v``. In latter
    case, if ``v`` is a :class:`chainerx.ndarray` object, then its view is
    returned.

Note:
    The argument ``v`` does not support array-like objects yet.

.. seealso:: :func:`numpy.diag`
""")

    _docs.set_doc(
        chainerx.diagflat,
        """diagflat(v, k=0, device=None)
Creates a diagonal array from the flattened input.

Args:
    v (~chainerx.ndarray): Array object.
    k (int): Index of diagonals. See :func:`chainerx.diag`.
    device (~chainerx.Device): Device on which the array is allocated.
        If omitted, :ref:`the default device <chainerx_device>` is chosen.

Returns:
    ~chainerx.ndarray: A 2-D diagonal array with the diagonal copied
    from ``v``.

Note:
    The argument ``v`` does not support array-like objects yet.

.. seealso:: :func:`numpy.diagflat`
""")


def _docs_indexing():
    _docs.set_doc(
        chainerx.take,
        """take(a, indices, axis)
Takes elements from an array along an axis.

Args:
    a (~chainerx.ndarray): Source array.
    indices (~chainerx.ndarray):
        The indices of the values to extract. When indices are out of bounds,
        they are wrapped around.
    axis (int): The axis over which to select values.

Returns:
    :func:`~chainerx.ndarray`: Output array.

Note:
    This function currently only supports indices of int64 array.

Note:
    This function currently does not support ``axis=None``

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``a``.

.. seealso:: :func:`numpy.take`
""")


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

    _docs.set_doc(
        chainerx.transpose,
        """transpose(a, axes=None)
Permutes the dimensions of an array.

Args:
    a (~chainerx.ndarray): Array to permute the dimensions.
    axes (tuple of ints): Permutation of the dimensions. This function reverses
        the shape by default.

Returns:
    ~chainerx.ndarray: A view of ``a`` with the dimensions permuted.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``a``.

.. seealso:: :func:`numpy.transpose`
""")

    _docs.set_doc(
        chainerx.broadcast_to,
        """broadcast_to(array, shape)
Broadcasts an array to a given shape.

Args:
    array (~chainerx.ndarray): Array to broadcast.
    shape (tuple of ints): The shape of the desired array.

Returns:
    ~chainerx.ndarray: Broadcasted view.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``array``.

.. seealso:: :func:`numpy.broadcast_to`
""")

    _docs.set_doc(
        chainerx.squeeze,
        """squeeze(a, axis=None)
Removes size-one axes from the shape of an array.

Args:
    a (~chainerx.ndarray): Array to be reshaped.
    axis (int or tuple of ints): Axes to be removed. This function removes all
        size-one axes by default. If one of the specified axes is not of size
        one, an exception is raised.

Returns:
    ~chainerx.ndarray: An array without (specified) size-one axes.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``a``.

.. seealso:: :func:`numpy.squeeze`
""")

    _docs.set_doc(
        chainerx.asscalar,
        """asscalar(a)
Converts an array of size 1 to its scalar equivalent.

Args:
    a (~chainerx.ndarray): Input array of size 1.

Returns:
    scalar: Scalar representation of ``a``. The output type is one of the \
Python scalar types (such as :class:`int` and :class:`float`) which \
corresponds to the dtype of ``a``.

.. seealso:: :func:`numpy.asscalar`
""")

    _docs.set_doc(
        chainerx.concatenate,
        """concatenate(arrays, axis=0)
Joins arrays along an axis.

Args:
    arrays (sequence of :class:`~chainerx.ndarray`\\ s): Arrays to be joined.
        All of these should have the same dimensionalities except the specified
        axis.
    axis (int): The axis to join arrays along.


Returns:
    ~chainerx.ndarray: Joined array.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input arrays in ``arrays``.

.. seealso:: :func:`numpy.concatenate`
""")

    _docs.set_doc(
        chainerx.stack,
        """stack(arrays, axis=0)
Stacks arrays along a new axis.

Args:
    arrays (sequence of :class:`~chainerx.ndarray`\\ s): Arrays to be stacked.
    axis (int): Axis along which the arrays are stacked.

Returns:
    ~chainerx.ndarray: Stacked array.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input arrays in ``arrays``.

.. seealso:: :func:`numpy.stack`
""")

    _docs.set_doc(
        chainerx.split,
        """split(ary, indices_or_sections, axis=0)
Splits an array into multiple sub arrays along a given axis.

Args:
    ary (~chainerx.ndarray): Array to split.
    indices_or_sections (int or sequence of ints): A value indicating how to
        divide the axis. If it is an integer, then is treated as the number of
        sections, and the axis is evenly divided. Otherwise, the integers
        indicate indices to split at. Note that a sequence on the device
        memory is not allowed.
    axis (int): Axis along which the array is split.

Returns:
    list of :class:`~chainerx.ndarray`\\ s: A list of sub arrays. Each array \
is a partial view of the input array.

Note:
    During backpropagation, this function propagates the gradients of the
    output arrays to the input array ``ary``.

.. seealso:: :func:`numpy.split`
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
    :class:`~chainerx.ndarray`:
        Returned array: :math:`y = max(\\{x_1, x_2\\})`.

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


def _docs_sorting():
    _docs.set_doc(
        chainerx.argmax,
        """argmax(a, axis=None)
Returns the indices of the maximum along an axis.

Args:
    a (~chainerx.ndarray): Array to take the indices of the maximum of.
    axis (None or int): Along which axis to compute the maximum. The flattened
        array is used by default.

Returns:
    :class:`~chainerx.ndarray`: The indices of the maximum of ``a``, along the
    axis if specified.

.. seealso:: :func:`numpy.argmax`
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
    _docs.set_doc(
        chainerx.conv,
        """conv(x, w, b=None, stride=1, pad=0, cover_all=False)
N-dimensional convolution.

This is an implementation of N-dimensional convolution which is generalized
two-dimensional convolution in ConvNets. It takes three arrays: the
input ``x``, the filter weight ``w`` and the bias vector ``b``.

Notation: here is a notation for dimensionalities.

- :math:`N` is the number of spatial dimensions.
- :math:`n` is the batch size.
- :math:`c_I` and :math:`c_O` are the number of the input and output
  channels, respectively.
- :math:`d_1, d_2, ..., d_N` are the size of each axis of the input's
  spatial dimensions, respectively.
- :math:`k_1, k_2, ..., k_N` are the size of each axis of the filters,
  respectively.
- :math:`l_1, l_2, ..., l_N` are the size of each axis of the output's
  spatial dimensions, respectively.
- :math:`p_1, p_2, ..., p_N` are the size of each axis of the spatial
  padding size, respectively.

Then the ``conv`` function computes correlations between filters
and patches of size :math:`(k_1, k_2, ..., k_N)` in ``x``.
Note that correlation here is equivalent to the inner product between
expanded tensors.
Patches are extracted at positions shifted by multiples of ``stride`` from
the first position ``(-p_1, -p_2, ..., -p_N)`` for each spatial axis.

Let :math:`(s_1, s_2, ..., s_N)` be the stride of filter application.
Then, the output size :math:`(l_1, l_2, ..., l_N)` is determined by the
following equations:

.. math::

   l_n = (d_n + 2p_n - k_n) / s_n + 1 \\ \\ (n = 1, ..., N)

If ``cover_all`` option is ``True``, the filter will cover the all
spatial locations. So, if the last stride of filter does not cover the
end of spatial locations, an addtional stride will be applied to the end
part of spatial locations. In this case, the output size is determined by
the following equations:

.. math::

   l_n = (d_n + 2p_n - k_n + s_n - 1) / s_n + 1 \\ \\ (n = 1, ..., N)

Args:
    x (:class:`~chainerx.ndarray`):
        Input array of shape :math:`(n, c_I, d_1, d_2, ..., d_N)`.
    w (:class:`~chainerx.ndarray`):
        Weight array of shape :math:`(c_O, c_I, k_1, k_2, ..., k_N)`.
    b (None or :class:`~chainerx.ndarray`):
        One-dimensional bias array with length :math:`c_O` (optional).
    stride (:class:`int` or :class:`tuple` of :class:`int` s):
        Stride of filter applications :math:`(s_1, s_2, ..., s_N)`.
        ``stride=s`` is equivalent to ``(s, s, ..., s)``.
    pad (:class:`int` or :class:`tuple` of :class:`int` s):
        Spatial padding width for input arrays
        :math:`(p_1, p_2, ..., p_N)`. ``pad=p`` is equivalent to
        ``(p, p, ..., p)``.
    cover_all (bool): If ``True``, all spatial locations are convoluted
        into some output pixels. It may make the output size larger.
        `cover_all` needs to be ``False`` if you want to use ``cuda`` backend.

Returns:
    ~chainerx.ndarray:
        Output array of shape :math:`(n, c_O, l_1, l_2, ..., l_N)`.

Note:

    In ``cuda`` backend, this function uses cuDNN implementation for its
    forward and backward computation.

Note:

    In ``cuda`` backend, this function has following limitations yet:

    - The ``cover_all=True`` option is not supported yet.
    - The ``dtype`` must be ``float32`` or ``float64`` (``float16`` is not
      supported yet.)

Note:

    During backpropagation, this function propagates the gradient of the
    output array to input arrays ``x``, ``w``, and ``b``.

.. seealso:: :func:`chainer.functions.convolution_nd`

.. admonition:: Example

    >>> n = 10
    >>> c_i, c_o = 3, 1
    >>> d1, d2, d3 = 30, 40, 50
    >>> k1, k2, k3 = 10, 10, 10
    >>> p1, p2, p3 = 5, 5, 5
    >>> x = chainerx.random.uniform(0, 1, (n, c_i, d1, d2, d3)).\
astype(np.float32)
    >>> x.shape
    (10, 3, 30, 40, 50)
    >>> w = chainerx.random.uniform(0, 1, (c_o, c_i, k1, k2, k3)).\
astype(np.float32)
    >>> w.shape
    (1, 3, 10, 10, 10)
    >>> b = chainerx.random.uniform(0, 1, (c_o)).astype(np.float32)
    >>> b.shape
    (1,)
    >>> s1, s2, s3 = 2, 4, 6
    >>> y = chainerx.conv(x, w, b, stride=(s1, s2, s3),\
 pad=(p1, p2, p3))
    >>> y.shape
    (10, 1, 16, 11, 9)
    >>> l1 = int((d1 + 2 * p1 - k1) / s1 + 1)
    >>> l2 = int((d2 + 2 * p2 - k2) / s2 + 1)
    >>> l3 = int((d3 + 2 * p3 - k3) / s3 + 1)
    >>> y.shape == (n, c_o, l1, l2, l3)
    True
    >>> y = chainerx.conv(x, w, b, stride=(s1, s2, s3),\
 pad=(p1, p2, p3), cover_all=True)
    >>> y.shape == (n, c_o, l1, l2, l3 + 1)
    True
""")

    _docs.set_doc(
        chainerx.conv_transpose,
        """conv_transpose(x, w, b=None, stride=1, pad=0, outsize=None)
N-dimensional transposed convolution.

This is an implementation of N-dimensional transposed convolution, which is
previously known as **deconvolution** in Chainer.

.. _Deconvolutional Networks: \
://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf

It takes three arrays: the input ``x``, the filter weight ``w``, and the
bias vector ``b``.

Notation: here is a notation for dimensionalities.

- :math:`N` is the number of spatial dimensions.
- :math:`n` is the batch size.
- :math:`c_I` and :math:`c_O` are the number of the input and output
  channels, respectively.
- :math:`d_1, d_2, ..., d_N` are the size of each axis of the input's
  spatial dimensions, respectively.
- :math:`k_1, k_2, ..., k_N` are the size of each axis of the filters,
  respectively.
- :math:`p_1, p_2, ..., p_N` are the size of each axis of the spatial
  padding size, respectively.
- :math:`s_1, s_2, ..., s_N` are the stride of each axis of filter
  application, respectively.

If ``outsize`` option is ``None``, the output size
:math:`(l_1, l_2, ..., l_N)` is determined by the following equations with
the items in the above list:

.. math::

   l_n = s_n (d_n - 1)  + k_n - 2 p_n \\ \\ (n = 1, ..., N)

If ``outsize`` option is given, the output size is determined by
``outsize``. In this case, the ``outsize`` :math:`(l_1, l_2, ..., l_N)`
must satisfy the following equations:

.. math::

   d_n = \\lfloor (l_n + 2p_n - k_n) / s_n \\rfloor + 1 \\ \\ \
   (n = 1, ..., N)

Args:
    x (:class:`~chainerx.ndarray`):
        Input array of shape :math:`(n, c_I, d_1, d_2, ..., d_N)`.
    w (:class:`~chainerx.ndarray`):
        Weight array of shape :math:`(c_I, c_O, k_1, k_2, ..., k_N)`.
    b (None or :class:`~chainerx.ndarray`):
        One-dimensional bias array with length :math:`c_O` (optional).
    stride (:class:`int` or :class:`tuple` of :class:`int` s):
        Stride of filter applications :math:`(s_1, s_2, ..., s_N)`.
        ``stride=s`` is equivalent to ``(s, s, ..., s)``.
    pad (:class:`int` or :class:`tuple` of :class:`int` s):
        Spatial padding width for input arrays
        :math:`(p_1, p_2, ..., p_N)`. ``pad=p`` is equivalent to
        ``(p, p, ..., p)``.
    outsize (None or :class:`tuple` of :class:`int` s):
        Expected output size of deconvolutional operation. It should be a
        tuple of ints :math:`(l_1, l_2, ..., l_N)`. Default value is
        ``None`` and the outsize is estimated by input size, stride and
        pad.

Returns:
    ~chainerx.ndarray:
        Output array of shape :math:`(n, c_O, l_1, l_2, ..., l_N)`.

Note:

    During backpropagation, this function propagates the gradient of the
    output array to input arrays ``x``, ``w``, and ``b``.

.. seealso:: :func:`chainer.functions.deconvolution_nd`

.. admonition:: Example

    **Example1**: the case when ``outsize`` is not given.

    >>> n = 10
    >>> c_i, c_o = 3, 1
    >>> d1, d2, d3 = 5, 10, 15
    >>> k1, k2, k3 = 10, 10, 10
    >>> p1, p2, p3 = 5, 5, 5
    >>> x = chainerx.random.uniform(0, 1, (n, c_i, d1, d2, d3)).\
astype(np.float32)
    >>> x.shape
    (10, 3, 5, 10, 15)
    >>> w = chainerx.random.uniform(0, 1, (c_i, c_o, k1, k2, k3)).\
astype(np.float32)
    >>> w.shape
    (3, 1, 10, 10, 10)
    >>> b = chainerx.random.uniform(0, 1, (c_o)).astype(np.float32)
    >>> b.shape
    (1,)
    >>> s1, s2, s3 = 2, 4, 6
    >>> y = chainerx.conv_transpose(x, w, b, stride=(s1, s2, s3), \
pad=(p1, p2, p3))
    >>> y.shape
    (10, 1, 8, 36, 84)
    >>> l1 = s1 * (d1 - 1) + k1 - 2 * p1
    >>> l2 = s2 * (d2 - 1) + k2 - 2 * p2
    >>> l3 = s3 * (d3 - 1) + k3 - 2 * p3
    >>> y.shape == (n, c_o, l1, l2, l3)
    True

    **Example2**: the case when ``outsize`` is given.

    >>> n = 10
    >>> c_i, c_o = 3, 1
    >>> d1, d2, d3 = 5, 10, 15
    >>> k1, k2, k3 = 10, 10, 10
    >>> p1, p2, p3 = 5, 5, 5
    >>> x = chainerx.array(np.random.uniform(0, 1, (n, c_i, d1, d2, d3)).\
astype(np.float32))
    >>> x.shape
    (10, 3, 5, 10, 15)
    >>> w = chainerx.array(np.random.uniform(0, 1, (c_i, c_o, k1, k2, k3)).\
astype(np.float32))
    >>> w.shape
    (3, 1, 10, 10, 10)
    >>> b = chainerx.array(np.random.uniform(0, 1, (c_o)).astype(np.float32))
    >>> b.shape
    (1,)
    >>> s1, s2, s3 = 2, 4, 6
    >>> l1, l2, l3 = 9, 38, 87
    >>> d1 == int((l1 + 2 * p1 - k1) / s1) + 1
    True
    >>> d2 == int((l2 + 2 * p2 - k2) / s2) + 1
    True
    >>> d3 == int((l3 + 2 * p3 - k3) / s3) + 1
    True
    >>> y = chainerx.conv_transpose(x, w, b, stride=(s1, s2, s3), \
pad=(p1, p2, p3), outsize=(l1, l2, l3))
    >>> y.shape
    (10, 1, 9, 38, 87)
    >>> y.shape == (n, c_o, l1, l2, l3)
    True
""")

    _docs.set_doc(
        chainerx.linear,
        """linear(x, W, b=None, n_batch_axis=1)
Linear function, or affine transformation.

It accepts two or three arguments: an input minibatch ``x``, a weight
matrix ``W``, and optionally a bias vector ``b``. It computes

.. math:: Y = xW^\\top + b.

Args:
    x (~chainerx.ndarray):
        Input array, which is a :math:`(s_1, s_2, ..., s_n)`-shaped array.
    W (~chainerx.ndarray):
        Weight variable of shape :math:`(M, N)`,
        where :math:`(N = s_{\\rm n\\_batch\\_axes} * ... * s_n)`.
    b (~chainerx.ndarray):
        Bias variable (optional) of shape :math:`(M,)`.
    n_batch_axes (int):
        The number of batch axes. The default is 1. The input variable is
        reshaped into (:math:`{\\rm n\\_batch\\_axes} + 1`)-dimensional
        tensor. This should be greater than 0.

Returns:
    :class:`~chainerx.ndarray`:
        Output array with shape of
        :math:`(s_1, ..., s_{\\rm n\\_batch\\_axes}, M)`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to input arrays ``x``, ``W`` and ``b``.
""")


def _docs_normalization():
    _docs.set_doc(
        chainerx.batch_norm,
        """batch_norm(x, gamma, beta, running_mean, running_var, eps=2e-5, decay=0.9, axis=None)
Batch normalization function.

It takes the input array ``x`` and two parameter arrays ``gamma`` and
``beta``. The parameter arrays must both have the same size.

Args:
    x (~chainerx.ndarray): Input array.
    gamma (~chainerx.ndarray): Scaling parameter of normalized data.
    beta (~chainerx.ndarray): Shifting parameter of scaled normalized data.
    running_mean (~chainerx.ndarray):
        Running average of the mean. This is a running average of
        the mean over several mini-batches using the decay parameter.
        The function takes a previous running average, and updates
        the array in-place by the new running average.
    running_var (~chainerx.ndarray):
        Running average of the variance. This is a running average of
        the variance over several mini-batches using the decay parameter.
        The function takes a previous running average, and updates
        the array in-place by the new running average.
    eps (float): Epsilon value for numerical stability.
    decay (float): Decay rate of moving average. It is used during training.
    axis (int, tuple of int or None):
        Axis over which normalization is performed. When axis is ``None``,
        the first axis is treated as the batch axis and will be reduced
        during normalization.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input arrays ``x``, ``gamma`` and ``beta``.

See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
      Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_
""")  # NOQA

    _docs.set_doc(
        chainerx.fixed_batch_norm,
        """fixed_batch_norm(x, gamma, beta, mean, var, eps=2e-5, axis=None)
Batch normalization function with fixed statistics.

This is a variant of :func:`~chainerx.batch_norm`, where the mean
and array statistics are given by the caller as fixed variables.

Args:
    x (~chainerx.ndarray): Input array.
    gamma (~chainerx.ndarray): Scaling parameter of normalized data.
    beta (~chainerx.ndarray): Shifting parameter of scaled normalized data.
    mean (~chainerx.ndarray): Shifting parameter of input.
    var (~chainerx.ndarray): Square of scaling parameter of input.
    eps (float): Epsilon value for numerical stability.
    axis (int, tuple of int or None):
        Axis over which normalization is performed. When axis is ``None``,
        the first axis is treated as the batch axis and will be reduced
        during normalization.

Note:
    During backpropagation, this function does not propagate gradients.
""")


def _docs_pooling():
    _docs.set_doc(
        chainerx.max_pool,
        """max_pool(x, ksize, stride=None, pad=0, cover_all=False)
Spatial max pooling function.

This acts similarly to :func:`~chainerx.conv`, but it computes the maximum
of input spatial patch for each channel without any parameter instead of
computing the inner products.

Args:
    x (~chainerx.ndarray): Input array.
    ksize (int or tuple of ints): Size of pooling window. ``ksize=k`` and
        ``ksize=(k, k, ..., k)`` are equivalent.
    stride (int or tuple of ints or None): Stride of pooling applications.
        ``stride=s`` and ``stride=(s, s, ..., s)`` are equivalent. If
        ``None`` is specified, then it uses same stride as the pooling
        window size.
    pad (int or tuple of ints): Spatial padding width for the input array.
        ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.
    cover_all (bool): If ``True``, all spatial locations are pooled into
        some output pixels. It may make the output size larger.

Returns:
    :class:`~chainerx.ndarray`:  Output array.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``. This function is only
    differentiable up to the second order.

.. note::
   In ``cuda`` backend, only 2 and 3 dim arrays are supported as ``x``
   because cuDNN pooling supports 2 and 3 spatial dimensions.
""")

    _docs.set_doc(
        chainerx.average_pool,
        """average_pool(x, ksize, stride=None, pad=0, pad_mode='ignore')
Spatial average pooling function.

This acts similarly to :func:`~chainerx.conv`, but it computes the average
of input spatial patch for each channel without any parameter instead of
computing the inner products.

Args:
    x (~chainerx.ndarray): Input array.
    ksize (int or tuple of ints): Size of pooling window. ``ksize=k`` and
        ``ksize=(k, k, ..., k)`` are equivalent.
    stride (int or tuple of ints or None): Stride of pooling applications.
        ``stride=s`` and ``stride=(s, s, ..., s)`` are equivalent. If
        ``None`` is specified, then it uses same stride as the pooling
        window size.
    pad (int or tuple of ints): Spatial padding width for the input array.
        ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.
    pad_mode ({'zero', 'ignore'}): Specifies how padded region is treated.

        * 'zero' -- the values in the padded region are treated as 0
        * 'ignore' -- padded region is ignored (default)

Returns:
    :class:`~chainerx.ndarray`:  Output array.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. note::
   In ``cuda`` backend, only 2 and 3 dim arrays are supported as ``x``
   because cuDNN pooling supports 2 and 3 spatial dimensions.
""")
