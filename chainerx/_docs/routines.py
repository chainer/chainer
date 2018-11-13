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
    _docs.set_doc(
        chainerx.conv,
        """conv(x, w, b=None, stride=1, pad=0, cover_all=False)
N-dimensional convolution.

This is an implementation of N-dimensional convolution which is generalized
two-dimensional convolution in ConvNets. It takes three variables: the
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

The N-dimensional convolution function is defined as follows.

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
    ~chainer.ndarray:
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

It takes three variables: the input ``x``, the filter weight ``w``, and the
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
        One-dimensional bias variable with length :math:`c_O` (optional).
    stride (:class:`int` or :class:`tuple` of :class:`int` s):
        Stride of filter applications :math:`(s_1, s_2, ..., s_N)`.
        ``stride=s`` is equivalent to ``(s, s, ..., s)``.
    pad (:class:`int` or :class:`tuple` of :class:`int` s):
        Spatial padding width for input arrays
        :math:`(p_1, p_2, ..., p_N)`. ``pad=p`` is equivalent to
        ``(p, p, ..., p)``.
    outsize (:class:`tuple` of :class:`int` s):
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


def _docs_normalization():
    pass


def _docs_pooling():
    pass
