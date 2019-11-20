import chainerx
from chainerx import _docs


def set_docs():
    _docs_creation()
    _docs_evaluation()
    _docs_indexing()
    _docs_linalg()
    _docs_logic()
    _docs_loss()
    _docs_manipulation()
    _docs_math()
    _docs_sorting()
    _docs_statistics()
    _docs_connection()
    _docs_normalization()
    _docs_pooling()
    _docs_rnn()


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
        chainerx.tri,
        """tri(N, M=None, k=0, dtype=float32, device=None)
Returns a 2-D array with ones at and below the given diagonal
and zeros elsewhere.

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
    ~chainerx.ndarray: A 2-D array with given diagonals filled ones at and
    below the given diagonal and zeros elsewhere.

.. seealso:: :func:`numpy.tri`
""")

    _docs.set_doc(
        chainerx.tril,
        """tril(m, k=0)
Lower triangle of an array.

Returns a copy of an array with elements above the k-th diagonal zeroed.

Args:
    m (~chainerx.ndarray): Input array.
    k (int): Index of the diagonal. Zero indicates the main diagonal,
        a positive index an upper diagonal, and a negative index a lower
        diagonal.

Returns:
    ~chainerx.ndarray: Lower triangle of ``m``.

.. seealso:: :func:`numpy.tril`
""")

    _docs.set_doc(
        chainerx.triu,
        """triu(m, k=0)
Upper triangle of an array.

Returns a copy of an array with elements below the k-th diagonal zeroed.

Args:
    m (~chainerx.ndarray): Input array.
    k (int): Index of the diagonal. Zero indicates the main diagonal,
        a positive index an upper diagonal, and a negative index a lower
        diagonal.

Returns:
    ~chainerx.ndarray: Upper triangle of ``m``.

.. seealso:: :func:`numpy.triu`
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

    _docs.set_doc(
        chainerx.meshgrid,
        """meshgrid(xi, indexing='xy')
Returns coordinate matrices from coordinate vectors.

Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector
fields over N-D grids, given one-dimensional coordinate arrays x1, x2,…, xn.

Args:
    xi (sequence of :class:`~chainerx.ndarray`\\ s): 1-D arrays
        representing the coordinates of a grid.
    indexing (str): {‘xy’, ‘ij’}, optional
        Cartesian (‘xy’, default) or matrix (‘ij’) indexing of output.

Returns:
    list of :class:`~chainerx.ndarray`\\ s: For vectors x1, x2,…, ‘xn’ with
    lengths Ni=len(xi), return (N1, N2, N3,...Nn) shaped arrays if
    indexing=’ij’ or (N2, N1, N3,...Nn) shaped arrays if indexing=’xy’
    with the elements of xi repeated to fill the matrix along the first
    dimension for x1, the second for x2 and so on.

.. seealso:: :func:`numpy.meshgrid`
""")


def _docs_evaluation():
    _docs.set_doc(
        chainerx.accuracy,
        """accuracy(y, t, ignore_label=None)
Computes multiclass classification accuracy of the minibatch.

Args:
    y (~chainerx.ndarray):
        Array whose (i, j, k, ...)-th element indicates the score of
        the class j at the (i, k, ...)-th sample.
        The prediction label :math:`\\hat t` is calculated by the formula
        :math:`\\hat t(i, k, ...) = \\operatorname{\\mathrm{argmax}}_j \
y(i, j, k, ...)`.
    t (~chainerx.ndarray):
        Array of ground truth labels.
    ignore_label (int or None): Skip calculating accuracy
        if the true label is ``ignore_label``.

Returns:
    :func:`~chainerx.ndarray`: A variable holding a scalar \
array of the accuracy.

Note:
    This function is non-differentiable.

.. seealso:: :func:`chainer.functions.accuracy`

.. admonition:: Example

    We show the most common case, when ``y`` is the two dimensional array.

    >>> y = chainerx.array([[0.1, 0.7, 0.2], # prediction label is 1
    ...                     [8.0, 1.0, 2.0], # prediction label is 0
    ...                     [-8.0, 1.0, 2.0], # prediction label is 2
    ...                     [-8.0, -1.0, -2.0]]) # prediction label is 1
    >>> t = chainerx.array([1, 0, 2, 1], chainerx.int32)
    >>> chainerx.accuracy(y, t) \
# 100% accuracy because all samples are correct
    array(1., shape=(), dtype=float64, device='native:0')
    >>> t = chainerx.array([1, 0, 0, 0], chainerx.int32)
    >>> chainerx.accuracy(y, t) \
# 50% accuracy because 1st and 2nd samples are correct
    array(0.5, shape=(), dtype=float64, device='native:0')
    >>> chainerx.accuracy(y, t, ignore_label=0) \
# 100% accuracy because of ignoring the 2nd, 3rd and 4th samples.
    array(1., shape=(), dtype=float64, device='native:0')
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
    mode (str): Specifies how out-of-bounds indices will behave.
        'raise' - raise an error
        'wrap' - wrap around
        'clip' - clip to the range

Returns:
    :func:`~chainerx.ndarray`: Output array.

Note:
    This function currently does not support ``axis=None``

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``a``.

Note:
   The default mode for the native backend is 'raise', while for the cuda
   backend is 'wrap' in order to prevent device synchronization.
   'raise' mode is currently not supported in the CUDA backend.

.. seealso:: :func:`numpy.take`
""")

    _docs.set_doc(
        chainerx.where,
        """where(condition, x, y)
Return elements chosen from ``x`` or ``y`` depending on condition.

Args:
    condition (~chainerx.ndarray): Where True, yield ``x``, otherwise
    yield ``y``.
    x (~chainerx.ndarray): Values from which to choose.
    y (~chainerx.ndarray): Values from which to choose.

Returns:
    :func:`~chainerx.ndarray`: An array with elements
    from ``x`` where condition is True, and elements from ``y`` elsewhere.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x`` and ``y``.

.. seealso:: :func:`numpy.where`
""")

    _docs.set_doc(
        chainerx.nonzero,
        """nonzero(a)
Return the indices of the elements that are non-zero.

Args:
    a (~chainerx.ndarray): Input array.

Returns:
    tuple of :func:`~chainerx.ndarray`: Indices of elements that are non-zero.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :func:`numpy.nonzero`
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

    _docs.set_doc(
        chainerx.linalg.solve,
        """solve(a, b)
Solves a linear matrix equation, or system of linear scalar equations.

It computes the exact solution of ``x`` in ``ax = b``,
where ``a`` is a square and full rank matrix,
``b`` can be a vector, or a rectangular matrix.
When ``b`` is matrix, its columns are treated as separate vectors
representing multiple right-hand sides.

Args:
    a (~chainerx.ndarray): Coefficient matrix.
    b (~chainerx.ndarray): "dependent variable" values.

Returns:
    :class:`~chainerx.ndarray`:
        Solution to the system ``ax = b``.
        Shape is identical to ``b``.

Note:
    The ``dtype`` must be ``float32`` or ``float64`` (``float16`` is not
    supported yet.)

.. seealso:: :func:`numpy.linalg.solve`
""")

    _docs.set_doc(
        chainerx.linalg.inv,
        """inv(a)
Computes the inverse of a matrix.

This function computes matrix ``a_inv`` from square matrix
``a`` such that ``dot(a, a_inv) = dot(a_inv, a) = eye(a.shape[0])``.

Args:
    a (~chainerx.ndarray): The matrix to be inverted.

Returns:
    :class:`~chainerx.ndarray`: The inverse of a matrix.

Note:
    The ``dtype`` must be ``float32`` or ``float64`` (``float16`` is not
    supported yet.)

.. seealso:: :func:`numpy.linalg.inv`
""")

    _docs.set_doc(
        chainerx.linalg.svd,
        """svd(a, full_matrices=True, compute_uv=True)
Singular Value Decomposition.

Factorizes the matrix ``a`` into two unitary matrices ``U`` and ``Vt``, and
a 1-D array ``s`` of singular values such that
``a == U * S * Vt``, where ``S`` is a suitably shaped matrix of zeros with
main diagonal ``s`` and ``*`` represents a dot product.

Args:
    a (~chainerx.ndarray): The input matrix with dimension ``(M, N)``.
    full_matrices (bool): If True, it returns u and v with dimensions
        ``(M, M)`` and ``(N, N)``. Otherwise, the dimensions of u and v
        are respectively ``(M, K)`` and ``(K, N)``, where
        ``K = min(M, N)``.
    compute_uv (bool): If False, only singular values are computed.

Returns:
    tuple of :class:`chainerx.ndarray`:
        A tuple of ``(U, s, Vt)`` such that ``a = U * diag(s) * Vt``.
        When ``compute_uv`` is False only singular values ``s`` are returned.

Note:
    * The ``dtype`` must be ``float32`` or ``float64`` (``float16`` is not
      supported yet.)
    * The SVD is commonly written as `a = U * diag(s) * V^T`.
      The ``Vt`` returned by this function is `V^T`.
    * During backpropagation, this function requires ``U`` and ``Vt`` computed,
      therefore differentiation does not work for ``compute_uv=False``.
    * Backpropagation is not implemented for ``full_matrices=True``.

.. seealso:: :func:`numpy.linalg.svd`
""")

    _docs.set_doc(
        chainerx.linalg.pinv,
        """pinv(a, rcond=1e-15)
Compute the (Moore-Penrose) pseudo-inverse of a matrix.

Calculate the generalized inverse of a matrix using its singular-value
decomposition (SVD) and including all large singular values.

Args:
    a (~chainerx.ndarray): The input matrix to be pseudo-inverted.
    rcond (float): Cutoff for small singular values.

Returns:
    :class:`~chainerx.ndarray`: The pseudo-inverse of ``a``.

Note:
    The ``dtype`` must be ``float32`` or ``float64`` (``float16`` is not
    supported yet.)

.. seealso:: :func:`numpy.linalg.pinv`
""")

    _docs.set_doc(
        chainerx.linalg.qr,
        """qr(a, mode='reduced')
Compute the qr factorization of a matrix.

Factor the matrix ``a`` as *qr*, where ``q`` is orthonormal and ``r`` is
upper-triangular.

Args:
    a (~chainerx.ndarray): Matrix to be factored.
    mode (str): The mode of decomposition.
        'reduced' : returns q, r with dimensions (M, K), (K, N) (default)
        'complete' : returns q, r with dimensions (M, M), (M, N)
        'r' : returns r only with dimensions (K, N)
        'raw' : returns h, tau with dimensions (N, M), (K,),
        where ``(M, N)`` is the shape of the input matrix and ``K = min(M, N)``

Returns:
    q (~chainerx.ndarray): A matrix with orthonormal columns.
    r (~chainerx.ndarray): The upper-triangular matrix.

Note:
    * The ``dtype`` must be ``float32`` or ``float64`` (``float16`` is not
      supported yet.)
    * Backpropagation is not implemented for non-square output matrix ``r``.
    * Backpropagation is not implemented for 'r' or 'raw' modes.

.. seealso:: :func:`numpy.linalg.qr`
""")

    _docs.set_doc(
        chainerx.linalg.cholesky,
        """cholesky(a)
Computes the Cholesky decomposition of a matrix.

Returns the Cholesky decomposition, :math:`A = L L^T`,
for the square matrix ``a``.

Args:
    a (~chainerx.ndarray): Symmetric positive-definite input matrix.

Returns:
    :class:`~chainerx.ndarray`: Output array. Cholesky factor of ``a``.

Note:
    The forward computation does not necessarily check if the input matrix is
    symmetric (e.g. the native backend relying on LAPACK does not). However,
    both the forward and the backward computations assume that it is and their
    results are unspecified otherwise. The computed gradient is always a
    symmetric matrix. More specifically, the gradient is computed as if the
    function is restricted to a Riemannian submanifold of
    :math:`R^{n \\times n}` consisting just of positive-definite symmetric
    matrices and is faithful to the mathematical definition of the Cholesky
    decomposition.

Note:
    * GPU implementation of the Cholesky decomposition routine is based on
      cuSOLVER library. Older versions (<10.1) of it might not raise an error
      for some non positive-definite matrices.
    * The ``dtype`` must be ``float32`` or ``float64`` (``float16`` is not
      supported yet.)

.. seealso:: :func:`numpy.linalg.cholesky`
""")

    _docs.set_doc(
        chainerx.linalg.eigh,
        """eigh(a, UPLO='L')
Compute the eigenvalues and eigenvectors of a real symmetric matrix.

Args:
    a (~chainerx.ndarray): Real symmetric matrix whose eigenvalues
        and eigenvectors are to be computed.
    UPLO (str): Specifies whether the calculation is done with the lower
        triangular part of a ('L', default) or the upper triangular part ('U').

Returns:
    tuple of :class:`~chainerx.ndarray`:
        Returns a tuple ``(w, v)``. ``w`` contains eigenvalues and
        ``v`` contains eigenvectors. ``v[:, i]`` is an eigenvector
        corresponding to an eigenvalue ``w[i]``.

Note:
    Although ``UPLO`` can be specified to ignore either the strictly lower or
    upper part of the input matrix, the backward computation assumes that the
    inputs is symmetric and the computed gradient is always a symmetric matrix
    with respect to ``UPLO``. More specifically, the gradient is computed as if
    the function is restricted to a Riemannian submanifold of
    :math:`R^{n \\times n}` consisting just of symmetric matrices and is
    faithful to the mathematical definition of the eigenvalue decomposition of
    symmetric matrices.

Note:
    The ``dtype`` must be ``float32`` or ``float64`` (``float16`` is not
    supported yet.)

.. seealso:: :func:`numpy.linalg.eigh`
""")

    _docs.set_doc(
        chainerx.linalg.eigvalsh,
        """eigvalsh(a, UPLO='L')
Compute the eigenvalues of a real symmetric matrix.

Main difference from eigh: the eigenvectors are not computed.

Args:
    a (~chainerx.ndarray): Real symmetric matrix whose eigenvalues
        and eigenvectors are to be computed.
    UPLO (str): Specifies whether the calculation is done with the lower
        triangular part of a (‘L’, default) or the upper triangular part (‘U’).
        (optional).

Returns:
    :class:`~chainerx.ndarray`: Returns eigenvalues as a vector.

Note:
    * The ``dtype`` must be ``float32`` or ``float64`` (``float16`` is not
      supported yet.)
    * Backpropagation requires eigenvectors and, therefore, is not implemented
      for this function. ``linalg.eigh`` should be used instead.

.. seealso:: :func:`numpy.linalg.eigvalsh`
""")


def _docs_logic():
    _docs.set_doc(
        chainerx.all,
        """all(x)
Test whether all array elements along a given axis evaluate to True.

Args:
    x (~chainerx.ndarray): Input array.
    axis (None or int or tuple of ints):
        Axis or axes along which AND reduction is performed.
        The flattened array is used by default.
    keepdims (bool):
        If this is set to ``True``, the reduced axes are left in the result
        as dimensions with size one.

Returns:
    :class:`~chainerx.ndarray`: Output array of type bool.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :data:`numpy.all`
""")

    _docs.set_doc(
        chainerx.any,
        """any(x)
Test whether any array element along a given axis evaluate to True.

Args:
    x (~chainerx.ndarray): Input array.
    axis (None or int or tuple of ints):
        Axis or axes along which OR reduction is performed.
        The flattened array is used by default.
    keepdims (bool):
        If this is set to ``True``, the reduced axes are left in the result
        as dimensions with size one.

Returns:
    :class:`~chainerx.ndarray`: Output array of type bool.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :data:`numpy.any`
""")

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
        chainerx.logical_and,
        """logical_and(x1, x2)
Returns an array of x1 AND x2 element-wise.

Args:
    x1 (~chainerx.ndarray): Input array.
    x2 (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Output array of type bool.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :data:`numpy.logical_and`
    """)

    _docs.set_doc(
        chainerx.logical_or,
        """logical_or(x1, x2)
Returns an array of x1 OR x2 element-wise.

Args:
    x1 (~chainerx.ndarray): Input array.
    x2 (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Output array of type bool.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :data:`numpy.logical_or`
    """)

    _docs.set_doc(
        chainerx.logical_xor,
        """logical_xor(x1, x2)
Returns an array of x1 XOR x2 element-wise.

Args:
    x1 (~chainerx.ndarray): Input array.
    x2 (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Output array of type bool.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :data:`numpy.logical_xor`
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


def _docs_loss():
    _docs.set_doc(
        chainerx.absolute_error,
        """Element-wise absolute error function.

Computes the element-wise absolute error :math:`L` between two inputs
:math:`x_1` and :math:`x_2` defined as follows.

.. math::
    L = |x_1 - x_2|

Args:
    x1 (~chainerx.ndarray): Input variable.
    x2 (~chainerx.ndarray): Input variable.

Returns:
    :class:`~chainerx.ndarray`: A variable holding an array representing
    the absolute error of two inputs.

.. seealso:: :func:`chainer.functions.absolute_error`
""")

    _docs.set_doc(
        chainerx.squared_error,
        """Element-wise squared error function.

Computes the element-wise squared error :math:`L` between two inputs
:math:`x_1` and :math:`x_2` defined as follows.

.. math::
    L = (x_1 - x_2)^2

Can be used to compute mean squared error by just calling `mean()`
on the output array.

Args:
    x0 (~chainerx.ndarray): Input variable.
    x1 (~chainerx.ndarray): Input variable.

Returns:
    :class:`~chainerx.ndarray`: A variable holding an array representing
    the squared error of two inputs.

.. seealso:: :func:`chainer.functions.squared_error`
""")

    _docs.set_doc(
        chainerx.huber_loss,
        """Element-wise Huber loss.

The Huber loss is similar to the squared error but is less sensitive to
outliers in the data. It is defined as

.. math::

    L_{\\delta}(a) = \\left \\{ \\begin{array}{cc}
    \\frac{1}{2} a^2 & {\\rm if~|a| \\leq \\delta} \\\\
    \\delta (|a| - \\frac{1}{2} \\delta) & {\\rm otherwise,}
    \\end{array} \\right.

where :math:`a = x - t` is the difference between the input :math:`x`
and the target :math:`t`.

See: `Huber loss - Wikipedia <https://en.wikipedia.org/wiki/Huber_loss>`_.

Args:
    x (~chainerx.ndarray): Input variable.
    t (~chainerx.ndarray): Target variable for regression.
    delta (float): Constant variable for Huber loss function as used in
        definition.

Returns:
    :class:`~chainerx.ndarray`:
        A variable object holding an array representing the Huber loss
        :math:`L_{\\delta}` of the two inputs.

.. seealso:: :func:`chainer.functions.huber_loss`
""")

    _docs.set_doc(
        chainerx.gaussian_kl_divergence,
        """Element-wise KL-divergence of Gaussian variables from the standard one.

Given two variable ``mean`` representing :math:`\\mu` and ``ln_var``
representing :math:`\\log(\\sigma^2)`, this function calculates
the element-wise KL-divergence between the given multi-dimensional
Gaussian :math:`N(\\mu, S)` and the standard Gaussian :math:`N(0, I)`

.. math::

   D_{\\mathbf{KL}}(N(\\mu, S) \\| N(0, I)),

where :math:`S` is a diagonal matrix such that :math:`S_{ii} = \\sigma_i^2`
and :math:`I` is an identity matrix.

Args:
    mean (~chainerx.ndarray):
        A variable representing mean of given
        gaussian distribution, :math:`\\mu`.
    ln_var (~chainerx.ndarray):
        A variable representing logarithm of
        variance of given gaussian distribution, :math:`\\log(\\sigma^2)`.

Returns:
    :class:`~chainerx.ndarray`:
        A variable representing KL-divergence between
        given gaussian distribution and the standard gaussian.

.. seealso:: :func:`chainer.functions.gaussian_kl_divergence`
""")

    _docs.set_doc(
        chainerx.sigmoid_cross_entropy,
        """sigmoid_cross_entropy(x1, x2)

Element-wise cross entropy loss for pre-sigmoid activations.

Args:
    x1 (~chainerx.ndarray): An array whose (i, j)-th element indicates the
        unnormalized log probability of the j-th unit at the i-th example.
    x2 (~chainerx.ndarray): An array whose (i, j)-th element indicates a signed
        integer vector of ground truth labels 0 or 1. If ``x2[i, j] == -1``,
        corresponding ``x1[i, j]`` is ignored. Loss is zero if all ground truth
        labels are -1.

Returns:
    :class:`~chainerx.ndarray`: An array of the cross entropy.

Note:
    During backpropagation, this function propagates the gradient of the output
    array to the input array ``x1`` only.
""")

    _docs.set_doc(
        chainerx.softmax_cross_entropy,
        """softmax_cross_entropy(x1, x2)

Element-wise cross entropy loss for pre-softmax activations.

Args:
    x1 (~chainerx.ndarray): An array whose element indicates unnormalized log
        probability: the first axis of the array represents the number of
        samples, and the second axis represents the number of classes.
    x2 (~chainerx.ndarray): A signed integer vector of ground truth labels. If
        ``x2[i] == -1``, corresponding ``x1[i]`` is ignored.

Returns:
    :class:`~chainerx.ndarray`: An array of the cross entropy.

Note:
    During backpropagation, this function propagates the gradient of the output
    array to the input array ``x1`` only.
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
        chainerx.ravel,
        """ravel(a)
Returns a flattened array.

Args:
    a (~chainerx.ndarray): Array to be flattened.

Returns:
    :class:`~chainerx.ndarray`: A flattened view of ``a`` if possible,
    otherwise a copy.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``a``.

.. seealso:: :func:`numpy.ravel`
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
        chainerx.hstack,
        """hstack(arrays)
Stack arrays in sequence horizontally (column wise).

Args:
    arrays (sequence of :class:`~chainerx.ndarray`\\ s): Arrays to be stacked.

Returns:
    ~chainerx.ndarray: Stacked array.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input arrays in ``arrays``.

.. seealso:: :func:`numpy.hstack`
""")

    _docs.set_doc(
        chainerx.vstack,
        """vstack(arrays)
Stack arrays in sequence vertically (row wise).

Args:
    arrays (sequence of :class:`~chainerx.ndarray`\\ s): Arrays to be stacked.

Returns:
    ~chainerx.ndarray: Stacked array.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input arrays in ``arrays``.

.. seealso:: :func:`numpy.vstack`
""")

    _docs.set_doc(
        chainerx.dstack,
        """dstack(arrays)
Stack arrays in sequence depth wise (along third axis).

Args:
    arrays (sequence of :class:`~chainerx.ndarray`\\ s): Arrays to be stacked.

Returns:
    ~chainerx.ndarray: Stacked array.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input arrays in ``arrays``.

.. seealso:: :func:`numpy.dstack`
""")

    _docs.set_doc(
        chainerx.atleast_2d,
        """atleast_2d(a)
View inputs as arrays with at least two dimensions.

Args:
    a (~chainerx.ndarray): Array.

Returns:
    ~chainerx.ndarray: An array with a.ndim >= 2.
    Copies are avoided where possible, and views with
    two or more dimensions are returned.

Note:
    * Arrays that already have two or more dimensions are preserved.
    * During backpropagation, this function propagates the gradient of the
      output array to the input arrays in ``a``.

.. seealso:: :func:`numpy.atleast_2d`
""")

    _docs.set_doc(
        chainerx.atleast_3d,
        """atleast_3d(a)
View inputs as arrays with at least three dimensions.

Args:
    a (~chainerx.ndarray): Array.

Returns:
    ~chainerx.ndarray: An array with a.ndim >= 3.
    Copies are avoided where possible, and views with
    three or more dimensions are returned.

Note:
    * Arrays that already have three or more dimensions are preserved.
    * During backpropagation, this function propagates the gradient of the
      output array to the input arrays in ``a``.

.. seealso:: :func:`numpy.atleast_3d`
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

    _docs.set_doc(
        chainerx.dsplit,
        """dsplit(ary, indices_or_sections)
Split array into multiple sub-arrays along the 3rd axis (depth).

Args:
    ary (~chainerx.ndarray): Array to split.
    indices_or_sections (int or sequence of ints): A value indicating how to
        divide the axis. If it is an integer, then is treated as the number of
        sections, and the axis is evenly divided. Otherwise, the integers
        indicate indices to split at. Note that a sequence on the device
        memory is not allowed.

Returns:
    list of :class:`~chainerx.ndarray`\\ s: A list of sub arrays. Each array \
is a partial view of the input array.

Note:
    During backpropagation, this function propagates the gradients of the
    output arrays to the input array ``ary``.

.. seealso:: :func:`numpy.dsplit`
""")

    _docs.set_doc(
        chainerx.vsplit,
        """vsplit(ary, indices_or_sections)
Splits an array into multiple sub-arrays vertically (row-wise).

Args:
    ary (~chainerx.ndarray): Array to split.
    indices_or_sections (int or sequence of ints): A value indicating how to
        divide the axis. If it is an integer, then is treated as the number of
        sections, and the axis is evenly divided. Otherwise, the integers
        indicate indices to split at. Note that a sequence on the device
        memory is not allowed.

Returns:
    list of :class:`~chainerx.ndarray`\\ s: A list of sub arrays. Each array \
is a partial view of the input array.

Note:
    During backpropagation, this function propagates the gradients of the
    output arrays to the input array ``ary``.

.. seealso:: :func:`numpy.vsplit`
""")

    _docs.set_doc(
        chainerx.hsplit,
        """hsplit(ary, indices_or_sections)
Split an array into multiple sub-arrays horizontally (column-wise).

Args:
    ary (~chainerx.ndarray): Array to split.
    indices_or_sections (int or sequence of ints): A value indicating how to
        divide the axis. If it is an integer, then is treated as the number of
        sections, and the axis is evenly divided. Otherwise, the integers
        indicate indices to split at. Note that a sequence on the device
        memory is not allowed.

Returns:
    list of :class:`~chainerx.ndarray`\\ s: A list of sub arrays. Each array \
is a partial view of the input array.

Note:
    During backpropagation, this function propagates the gradients of the
    output arrays to the input array ``ary``.

.. seealso:: :func:`numpy.hsplit`
""")

    _docs.set_doc(
        chainerx.swapaxes,
        """swapaxes(a, axis1, axis2)
Interchange two axes of an array.

Args:
    a (~chainerx.ndarray): Array to swapaxes.
    axis1 (int): First Axis
    axis2 (int): Second Axis

Returns:
    ~chainerx.ndarray: Swaped array.

Note:
    * Output array is a view of the input array.
    * During backpropagation, this function propagates the gradients of the
      output arrays to the input array ``a``.


.. seealso:: :func:`numpy.swapaxes`
""")

    _docs.set_doc(
        chainerx.repeat,
        """repeat(a, repeats, axis=None)
Constructs an array by repeating a given array.

Args:
    a (~chainerx.ndarray): Array to repeat.
    repeats (int or tuple of ints): The number of times which each
        element of a is repeated.
    axis (int): The axis along which to repeat values.

Returns:
    ~chainerx.ndarray: The repeated output array.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``a``.

.. seealso:: :func:`numpy.repeat`
""")

    _docs.set_doc(
        chainerx.expand_dims,
        """expand_dims(a, axis)
Expand the shape of an array.

Args:
    a (~chainerx.ndarray): Input Array.
    axis (int): Position in the expanded axes where the new axis is placed.

Returns:
    ~chainerx.ndarray: Output array.

Note:
    * Output array may or may not be a view of the input array.
    * During backpropagation, this function propagates the gradients of the
      output arrays to the input array ``a``.


.. seealso:: :func:`numpy.expand_dims`
""")

    _docs.set_doc(
        chainerx.flip,
        """flip(m, axis)
Reverse the order of elements in an array along the given axis.

Args:
    m (~chainerx.ndarray): Input Array.
    axis (int or tuple of ints): Axis or axes along which to flip over.
    The default, axis=None, will flip over all of the axes of the input array.
    If axis is negative it counts from the last to the first axis.
    If axis is a tuple of ints, flipping is performed on all of the
    axes specified in the tuple.

Returns:
    ~chainerx.ndarray: A view of m with the entries of axis reversed.
    Since a view is returned, this operation is done in constant time.

Note:
    * Output array is a view of the input array.
    * During backpropagation, this function propagates the gradients of the
      output arrays to the input array ``m``.


.. seealso:: :func:`numpy.flip`
""")

    _docs.set_doc(
        chainerx.fliplr,
        """fliplr(m)
Flip array in the left/right direction.

Args:
    m (~chainerx.ndarray): Input Array.

Returns:
    ~chainerx.ndarray: A view of m with the columns reversed.
    Since a view is returned, this operation is done in constant time.

Note:
    * Output array is a view of the input array.
    * During backpropagation, this function propagates the gradients of the
      output arrays to the input array ``m``.


.. seealso:: :func:`numpy.fliplr`
""")

    _docs.set_doc(
        chainerx.flipud,
        """flipud(m)
Flip array in the up/down direction.

Args:
    m (~chainerx.ndarray): Input Array.

Returns:
    ~chainerx.ndarray: A view of m with the rows reversed.
    Since a view is returned, this operation is done in constant time.

Note:
    * Output array is a view of the input array.
    * During backpropagation, this function propagates the gradients of the
      output arrays to the input array ``m``.


.. seealso:: :func:`numpy.flipud`
""")

    _docs.set_doc(
        chainerx.moveaxis,
        """moveaxis(a, source, destination)
Move axes of an array to new positions.

Other axes remain in their original order.

Args:
    a (~chainerx.ndarray): Input Array.
    source (int or tuple of ints): Original positions of the axes to move.
    These must be unique.
    destintation (int or tuple of ints): Destination positions for each of
    the original axes. These must also be unique.

Returns:
    ~chainerx.ndarray: Array with moved axes. This array is a view of the
    input array.

Note:
    * During backpropagation, this function propagates the gradients of the
      output arrays to the input array ``a``.


.. seealso:: :func:`numpy.moveaxis`
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

.. seealso:: :data:`numpy.maximum`
""")

    _docs.set_doc(
        chainerx.minimum,
        """minimum(x1, x2)
Minimum arguments, element-wise.

Args:
    x1 (~chainerx.ndarray or scalar): Input array.
    x2 (~chainerx.ndarray or scalar): Input array.

Returns:
    :class:`~chainerx.ndarray`:
        Returned array: :math:`y = min(\\{x_1, x_2\\})`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input arrays ``x1`` and ``x2``.

.. seealso:: :data:`numpy.minimum`
""")

    _docs.set_doc(
        chainerx.remainder,
        """remainder(x1, x2)
Return element-wise remainder of division.

Args:
    x1 (~chainerx.ndarray or scalar): Input array.
    x2 (~chainerx.ndarray or scalar): Input array.

Returns:
    :class:`~chainerx.ndarray`:
        Returned array: The element-wise remainder of
        the quotient ``floor_divide(x1, x2)``.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input arrays ``x1`` and ``x2``.

.. seealso:: :data:`numpy.remainder`
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
        chainerx.log10,
        """log10(x)
Base 10 logarithm, element-wise.

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\log_{10} x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.log10`
""")

    _docs.set_doc(
        chainerx.log2,
        """log2(x)
Base 2 logarithm, element-wise.

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\log_{2} x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.log2`
""")

    _docs.set_doc(
        chainerx.log1p,
        """log1p(x)
Natural logarithm of one plus the input, element-wise.

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\log(1 + x)`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.log1p`
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
        chainerx.square,
        """square(x)
Returns the element-wise square of the input.

Args:
    x (~chainerx.ndarray or scalar): Input data

Returns:
    ~chainerx.ndarray: Returned array: :math:`y = x * x`.
    A scalar is returned if ``x`` is a scalar.

Note:
    During backpropagation, this function propagates the gradient
    of the output array to the input array ``x``.

.. seealso:: :data:`numpy.square`
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
        chainerx.sinh,
        """sinh(x)
Hyperbolic Sine, element-wise

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\sinh x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.sinh`
""")

    _docs.set_doc(
        chainerx.cosh,
        """cosh(x)
Hyperbolic Cosine, element-wise

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\cosh x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.cosh`
""")

    _docs.set_doc(
        chainerx.tanh,
        """tanh(x)
Element-wise hyperbolic tangent function.

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
        chainerx.sigmoid,
        """sigmoid(x)
Element-wise sigmoid logistic function.

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array:
    :math:`f(x) = (1 + \\exp(-x))^{-1}`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :func:`chainer.functions.sigmoid`
""")

    _docs.set_doc(
        chainerx.sin,
        """sin(x)
Sine, element-wise

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\sin x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.sin`
""")

    _docs.set_doc(
        chainerx.cos,
        """cos(x)
Cosine, element-wise

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\cos x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.cos`
""")

    _docs.set_doc(
        chainerx.ceil,
        """ceil(x)
Return the ceiling of the input, element-wise..

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: The ceiling of each element in array.

.. seealso:: :data:`numpy.ceil`
""")

    _docs.set_doc(
        chainerx.tan,
        """tan(x)
Tangent, element-wise

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\tan x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.tan`
""")

    _docs.set_doc(
        chainerx.relu,
        """Rectified Linear Unit function.
Args:
    x (~chainerx.ndarray): Input array.
Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\max (0, x)`.
Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.
""")

    _docs.set_doc(
        chainerx.tree_lstm,
        """tree_lstm(*inputs)
TreeLSTM unit as an activation function.

This function implements TreeLSTM units both for
N-ary TreeLSTM and Child-Sum TreeLSTM.
Let the children cell states
:math:`c_{\\text{1}}, c_{\\text{2}}, \\dots, c_{\\text{N}}`,
and the incoming signal :math:`x`.
First, the incoming signal :math:`x` is split into (3 + N) arrays
:math:`a, i, o, f_{\\text{1}}, f_{\\text{2}}, ..., f_{\\text{N}}`
of the same shapes along the second axis.
It means that :math:`x` 's second axis must have (3 + N) times
of the length of each :math:`c_{n}`.
The splitted input signals are corresponding to

    - :math:`a` : sources of cell input
    - :math:`i` : sources of input gate
    - :math:`o` : sources of output gate
    - :math:`f_{n}` : sources of forget gate for n-th ary

Second, it computes outputs as

.. math::
    c &= \\tanh(a) \\text{sigmoid}(i) \\\\
      & + c_{\\text{1}} \\text{sigmoid}(f_{\\text{1}}), \\\\
      & + c_{\\text{2}} \\text{sigmoid}(f_{\\text{2}}), \\\\
      & + ..., \\\\
      & + c_{\\text{N}} \\text{sigmoid}(f_{\\text{N}}), \\\\
    h &= \\tanh(c) \\text{sigmoid}(o).

These are returned as a tuple of (N + 1) variables.

Args:
    inputs (list of :class:`~chainerx.array`): Variable arguments which
        include all cell vectors from child-nodes, and an input vector.
        Each of the cell vectors and the input vector is
        :class:`~chainerx.array`.
        The input vector must have the second dimension whose size
        is (N + 3) times of that of each cell,
        where N denotes the total number of cells.

Returns:
    tuple: Two :class:`~chainerx.array` objects ``c`` and ``h``. ``c`` is
    the updated cell state. ``h`` indicates the outgoing signal.

See the papers for details: `Improved Semantic Representations From
Tree-Structured Long Short-Term Memory Networks
<https://www.aclweb.org/anthology/P15-1150>`_ and
`A Fast Unified Model for Parsing and Sentence Understanding
<https://arxiv.org/pdf/1603.06021.pdf>`_.
Tai et al.'s N-Ary TreeLSTM is little extended in
Bowman et al., and this link is based on
the variant by Bowman et al.
Specifically, eq. 10 in Tai et al. only has one :math:`W` matrix
to be applied to :math:`x`, consistently for all children.
On the other hand, Bowman et al.'s model has multiple matrices,
each of which affects the forget gate for each child's cell individually.

.. admonition:: Example

    Assuming ``y`` is the current input signal, ``c`` is the previous cell
    state, and ``h`` is the previous output signal from an
    :meth:`~chainerx.tree_lstm` function.
    Each of ``y``, ``c`` and ``h`` has ``n_units`` channels.
    Using 2-ary (binary) TreeLSTM,

    most typical preparation of ``x`` is

    >>> c1 = chainerx.ones((4, 10), dtype = chainerx.float32)
    >>> c2 = chainerx.ones((4, 10), dtype = chainerx.float32)
    >>> x = chainerx.ones((4, 50), dtype = chainerx.float32)
    >>> c, h = chainerx.tree_lstm(c1, c2, x)
    """)

    _docs.set_doc(
        chainerx.slstm,
        """slstm(c_prev1, c_prev2, x1, x2)
S-LSTM units as an activation function.

This function implements S-LSTM unit. It is an extension of LSTM unit
applied to tree structures.
The function is applied to binary trees. Each node has two child nodes.
It gets four arguments, previous cell states ``c_prev1`` and ``c_prev2``,
and input arrays ``x1`` and ``x2``.
First both input arrays ``x1`` and ``x2`` are split into eight arrays
:math:`a_1, i_1, f_1, o_1`, and :math:`a_2, i_2, f_2, o_2`. They have the
same shape along the second axis.
It means that ``x1`` and ``x2`` 's second axis must have 4 times
the length of ``c_prev1`` and ``c_prev2``.
The split input arrays are corresponding to

    - :math:`a_i` : sources of cell input
    - :math:`i_i` : sources of input gate
    - :math:`f_i` : sources of forget gate
    - :math:`o_i` : sources of output gate

It computes the updated cell state ``c`` and the outgoing signal
``h`` as.

.. math::
    c &= \\tanh(a_1 + a_2) \\sigma(i_1 + i_2)
       + c_{\\text{prev}1} \\sigma(f_1)
       + c_{\\text{prev}2} \\sigma(f_2), \\\\
    h &= \\tanh(c) \\sigma(o_1 + o_2),

where :math:`\\sigma` is the elementwise sigmoid function.
The function returns ``c`` and ``h`` as a tuple.

Args:
    c_prev1 (:class:`~chainerx.array`):
        Variable that holds the previous cell state of the first child
        node. The cell state should be a zero array or the output of
        the previous call of LSTM.
    c_prev2 (:class:`~chainerx.array`):
        Variable that holds the previous cell state of the second child
        node.
    x1 (:class:`~chainerx.array`):
        Variable that holds the sources of cell input, input gate, forget
        gate and output gate from the first child node. It must have the
        second dimension whose size is four times of that of the cell
        state.
    x2 (:class:`~chainerx.array`):
        Variable that holds the input sources from the second child node.

Returns:
    tuple: Two :class:`~chainerx.array` objects ``c`` and ``h``. ``c`` is
    the cell state. ``h`` indicates the outgoing signal.

See detail in paper: `Long Short-Term Memory Over Tree Structures
<https://arxiv.org/abs/1503.04881>`_.

.. admonition:: Example

    Assuming ``c1``, ``c2`` is the previous cell state of children,
    and ``h1``, ``h2`` is the previous outgoing signal from children.
    Each of ``c1``, ``c2``, ``h1`` and ``h2`` has ``n_units`` channels.
    Most typical preparation of ``x1``, ``x2`` is:

    >>> n_units = 100
    >>> c1 = chainerx.ones((1, n_units), np.float32)
    >>> c2 = chainerx.ones((1, n_units), np.float32)
    >>> x1 = chainerx.ones((1, 4 * n_units), chainerx.float32)
    >>> x2 = chainerx.ones((1, 4 * n_units), chainerx.float32)
    >>> c, h = chainerx.slstm(c1, c2, x1, x2)
    """)

    _docs.set_doc(
        chainerx.arcsin,
        """arcsin(x)
Inverse sine, element-wise

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\arcsin x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.arcsin`
""")

    _docs.set_doc(
        chainerx.arccos,
        """arccos(x)
Trigonometric inverse cosine, element-wise

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\arccos x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.arccos`
""")

    _docs.set_doc(
        chainerx.arctan,
        """arctan(x)
Trigonometric inverse tangent, element-wise

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\arctan x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.arctan`
""")

    _docs.set_doc(
        chainerx.arctan2,
        """arctan2(x1, x2)
Element-wise arc tangent of :math:`\\frac{x_1}{x_2}` choosing the quadrant
correctly.

Args:
    x1 (~chainerx.ndarray): Input array.
    x2 (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returns an array where each element
    represents :math:`\\theta` in the range :math:`[-\\pi, \\pi]`, such
    that :math:`x_1 = r \\sin(\\theta)` and :math:`x_2 = r \\cos(\\theta)`
    for some :math:`r > 0`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x1`` and/or ``x2``.

.. seealso:: :data:`numpy.arctan2`
""")

    _docs.set_doc(
        chainerx.arcsinh,
        """arcsinh(x)
Inverse hyperbolic sine, element-wise

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\arcsinh x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.arcsinh`
""")

    _docs.set_doc(
        chainerx.arccosh,
        """arccosh(x)
Inverse hypberbolic inverse cosine, element-wise

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = \\arccosh x`.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``x``.

.. seealso:: :data:`numpy.arccosh`
""")

    _docs.set_doc(
        chainerx.fabs,
        """fabs(x)
Compute the absolute values element-wise.
Args:
    x (~chainerx.ndarray): Input array.
Returns:
    :class:`~chainerx.ndarray`: The absolute values of x, the returned values
    are always floats.
.. seealso:: :data:`numpy.fabs`
""")

    _docs.set_doc(
        chainerx.sign,
        """sign(x)
Returns an element-wise indication of the sign of a number.
The sign function returns :math:`-1 if x < 0, 0 if x==0, 1 if x > 0`.
``nan`` is returned for ``nan`` inputs.
Args:
    x (~chainerx.ndarray): Input array.
Returns:
    :class:`~chainerx.ndarray`: The sign of x.
.. seealso:: :data:`numpy.sign`
""")

    _docs.set_doc(
        chainerx.floor,
        """floor(x)
Return the floor of the input, element-wise.
Args:
    x (~chainerx.ndarray): Input array.
Returns:
    :class:`~chainerx.ndarray`: The floor of each element in array.
.. seealso:: :data:`numpy.floor`
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
        chainerx.isfinite,
        """isfinite(x)
Test element-wise for finiteness (not infinity or not Not a Number).

Args:
    x (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: True where x is not positive infinity,
    negative infinity, or NaN; false otherwise.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :data:`numpy.isfinite`
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

    _docs.set_doc(
        chainerx.bitwise_and,
        """bitwise_and(x1, x2)
Compute the bit-wise AND of two arrays element-wise.

Args:
    x1 (~chainerx.ndarray or scalar): Input array of integers.
    x2 (~chainerx.ndarray or scalar): Input array of integers.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = x_1 \\& x_2`

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :data:`numpy.bitwise_and`
""")

    _docs.set_doc(
        chainerx.bitwise_or,
        """bitwise_or(x1, x2)
Compute the bit-wise OR of two arrays element-wise.

Args:
    x1 (~chainerx.ndarray or scalar): Input array of integers.
    x2 (~chainerx.ndarray or scalar): Input array of integers.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = x_1 | x_2`

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :data:`numpy.bitwise_or`
""")

    _docs.set_doc(
        chainerx.bitwise_xor,
        """bitwise_xor(x1, x2)
Compute the bit-wise XOR of two arrays element-wise.

Args:
    x1 (~chainerx.ndarray or scalar): Input array of integers.
    x2 (~chainerx.ndarray or scalar): Input array of integers.

Returns:
    :class:`~chainerx.ndarray`: Returned array: :math:`y = x_1 \\oplus x_2`

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :data:`numpy.bitwise_xor`
""")

    _docs.set_doc(
        chainerx.left_shift,
        """left_shift(x1, x2)
Shift the bits of an integer to the left.

Args:
    x1 (~chainerx.ndarray or scalar): Input array of integers.
    x2 (~chainerx.ndarray or scalar): Input array of integers.

Returns:
    :class:`~chainerx.ndarray`: Return `x1` with bits shifted `x2` times to the left.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :data:`numpy.left_shift`
""")  # NOQA

    _docs.set_doc(
        chainerx.right_shift,
        """right_shift(x1, x2)
Shift the bits of an integer to the right.

Args:
    x1 (~chainerx.ndarray or scalar): Input array of integers.
    x2 (~chainerx.ndarray or scalar): Input array of integers.

Returns:
    :class:`~chainerx.ndarray`: Return `x1` with bits shifted `x2` times to the right.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :data:`numpy.right_shift`
""")  # NOQA


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

    _docs.set_doc(
        chainerx.argmin,
        """argmin(a, axis=None)
Returns the indices of the minimum along an axis.

Args:
    a (~chainerx.ndarray): Array to take the indices of the minimum of.
    axis (None or int): Along which axis to compute the minimum. The flattened
        array is used by default.

Returns:
    :class:`~chainerx.ndarray`: The indices of the minimum of ``a``, along the
    axis if specified.

.. seealso:: :func:`numpy.argmin`
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

    _docs.set_doc(
        chainerx.amin,
        """amin(a, axis=None, keepdims=False)
Returns the minimum of an array or the minimum along an axis.

Note:
    When at least one element is NaN, the corresponding min value will be NaN.

Args:
    a (~chainerx.ndarray): Array to take the minimum.
    axis (None or int or tuple of ints): Along which axis to take the minimum.
        The flattened array is used by default.
        If this is a tuple of ints, the minimum is selected over multiple
        axes, instead of a single axis or all the axes.
    keepdims (bool): If ``True``, the axis is remained as an axis of size one.

Returns:
    :class:`~chainerx.ndarray`: The minimum of ``a``, along the axis if
    specified.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``a``.

.. seealso:: :func:`numpy.amin`
""")

    _docs.set_doc(
        chainerx.mean,
        """mean(a, axis=None, keepdims=False)
Compute the arithmetic mean along the specified axis.

Returns the average of the array elements. The average is taken over the
flattened array by default, otherwise over the specified axis.

Args:
    a (~chainerx.ndarray): Array to take the mean of.
    axis (None or int or tuple of ints): Along which axis or axes to compute
    the mean. The flattened array is used by default.
    keepdims (bool): If this is set to True, the axes which are reduced are
    left in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

Returns:
    :class:`~chainerx.ndarray`: The mean of ``a``, along the axis or axes if
    specified.

.. seealso:: :func:`numpy.mean`
""")

    _docs.set_doc(
        chainerx.var,
        """var(a, axis=None, keepdims=False)
Compute the arithmetic var along the specified axis.

Returns the var of the array elements. The var is taken over the flattened
array by default, otherwise over the specified axis.

Args:
    a (~chainerx.ndarray): Array to take the var of.
    axis (None or int or tuple of ints): Along which axis or axes to compute
    the var. The flattened array is used by default.
    keepdims (bool): If this is set to True, the axes which are reduced are
    left in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

Returns:
    :class:`~chainerx.ndarray`: The var of ``a``, along the axis or axes if
    specified.

.. seealso:: :func:`numpy.var`
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
end of spatial locations, an additional stride will be applied to the end
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

    _docs.set_doc(
        chainerx.lstm,
        """lstm(c_prev, x)
Long Short-Term Memory units as an activation function.

This function implements LSTM units with forget gates. Let the previous
cell state ``c_prev`` and the input array ``x``.
First, the input array ``x`` is split into four arrays
:math:`a, i, f, o` of the same shapes along the second axis. It means that
``x`` 's second axis must have 4 times the ``c_prev`` 's second axis.
The split input arrays are corresponding to:

    - :math:`a` : sources of cell input
    - :math:`i` : sources of input gate
    - :math:`f` : sources of forget gate
    - :math:`o` : sources of output gate

Second, it computes the updated cell state ``c`` and the outgoing signal
``h`` as

.. math::
    c &= \\tanh(a) \\sigma(i)
       + c_{\\text{prev}} \\sigma(f), \\\\
    h &= \\tanh(c) \\sigma(o),

where :math:`\\sigma` is the elementwise sigmoid function.
These are returned as a tuple of two variables.
This function supports variable length inputs. The mini-batch size of
the current input must be equal to or smaller than that of the previous
one. When mini-batch size of ``x`` is smaller than that of ``c``, this
function only updates ``c[0:len(x)]`` and doesn't change the rest of ``c``,
``c[len(x):]``. So,
please sort input sequences in descending order of lengths before
applying the function.

Args:
    c_prev (:class:`~chainerx.array`):
        Variable that holds the previous cell state. The cell state
        should be a zero array or the output of the previous call of LSTM.
    x (:class:`~chainer.array`):
        Variable that holds the sources of cell input, input gate, forget
        gate and output gate. It must have the second dimension whose size
        is four times of that of the cell state.

Returns:
    tuple: Two :class:`~chainerx.array` objects ``c`` and ``h``.
    ``c`` is the updated cell state. ``h`` indicates the outgoing signal.

See the original paper proposing LSTM with forget gates:
`Long Short-Term Memory in Recurrent Neural Networks
<http://www.felixgers.de/papers/phd.pdf>`_.

.. admonition:: Example

    Assuming ``y`` is the current incoming signal, ``c`` is the previous
    cell state, and ``h`` is the previous outgoing signal from an ``lstm``
    function. Each of ``y``, ``c`` and ``h`` has ``n_units`` channels.
    Most typical preparation of ``x`` is

    >>> n_units = 100
    >>> c_prev = chainerx.zeros((1, n_units), chainerx.float32)
    >>> x = chainerx.zeros((1, 4 * n_units), chainerx.float32)
    >>> c, h = chainerx.lstm(c_prev, x)

    It corresponds to calculate the input array ``x``, or the input
    sources :math:`a, i, f, o`, from the current incoming signal ``y`` and
    the previous outgoing signal ``h``. Different parameters are used for
    different kind of input sources.
""")


def _docs_normalization():
    _docs.set_doc(
        chainerx.batch_norm,
        """batch_norm(x, gamma, beta, running_mean, running_var, eps=2e-5, \
decay=0.9, axis=None)
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
""")

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


def _docs_rnn():
    _docs.set_doc(
        chainerx.n_step_lstm,
        """n_step_lstm(n_layers, hx, cx, ws, bs, xs)
    Stacked Uni-directional Long Short-Term Memory function.

This function calculates stacked Uni-directional LSTM with sequences.
This function gets an initial hidden state :math:`h_0`, an initial cell
state :math:`c_0`, an input sequence :math:`x`, weight matrices :math:`W`,
and bias vectors :math:`b`.
This function calculates hidden states :math:`h_t` and :math:`c_t` for each
time :math:`t` from input :math:`x_t`.

.. math::
   i_t &= \\sigma(W_0 x_t + W_4 h_{t-1} + b_0 + b_4) \\\\
   f_t &= \\sigma(W_1 x_t + W_5 h_{t-1} + b_1 + b_5) \\\\
   o_t &= \\sigma(W_2 x_t + W_6 h_{t-1} + b_2 + b_6) \\\\
   a_t &= \\tanh(W_3 x_t + W_7 h_{t-1} + b_3 + b_7) \\\\
   c_t &= f_t \\cdot c_{t-1} + i_t \\cdot a_t \\\\
   h_t &= o_t \\cdot \\tanh(c_t)

As the function accepts a sequence, it calculates :math:`h_t` for all
:math:`t` with one call. Eight weight matrices and eight bias vectors are
required for each layer. So, when :math:`S` layers exist, you need to
prepare :math:`8S` weight matrices and :math:`8S` bias vectors.
If the number of layers ``n_layers`` is greater than :math:`1`, the input
of the ``k``-th layer is the hidden state ``h_t`` of the ``k-1``-th layer.
Note that all input variables except the first layer may have different
shape from the first layer.

Args:
    n_layers(int): The number of layers.
    hx (:class:`~chainerx.array`):
        Variable holding stacked hidden states.
        Its shape is ``(S, B, N)`` where ``S`` is the number of layers and
        is equal to ``n_layers``, ``B`` is the mini-batch size, and ``N``
        is the dimension of the hidden units.
    cx (:class:`~chainerx.array`): Variable holding stacked cell states.
        It has the same shape as ``hx``.
    ws (list of list of :class:`~chainerx.array`): Weight matrices.
        ``ws[i]`` represents the weights for the i-th layer.
        Each ``ws[i]`` is a list containing eight matrices.
        ``ws[i][j]`` corresponds to :math:`W_j` in the equation.
        Only ``ws[0][j]`` where ``0 <= j < 4`` are ``(N, I)``-shaped as
        they are multiplied with input variables, where ``I`` is the size
        of the input and ``N`` is the dimension of the hidden units. All
        other matrices are ``(N, N)``-shaped.
    bs (list of list of :class:`~chainerx.array`): Bias vectors.
        ``bs[i]`` represents the biases for the i-th layer.
        Each ``bs[i]`` is a list containing eight vectors.
        ``bs[i][j]`` corresponds to :math:`b_j` in the equation.
        The shape of each matrix is ``(N,)`` where ``N`` is the dimension
        of the hidden units.
    xs (list of :class:`~chainerx.array`):
        A list of :class:`~chainerx.array`
        holding input values. Each element ``xs[t]`` holds input value
        for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is the
        mini-batch size for time ``t``.
        When sequences has different lengths, they must be
        sorted in descending order of their lengths.
        So ``xs`` needs to satisfy
        ``xs[t].shape[0] >= xs[t + 1].shape[0]``.

Returns:
    tuple: This function returns a tuple containing three elements,
    ``hy``, ``cy`` and ``ys``.

    - ``hy`` is an updated hidden states whose shape is the same as
      ``hx``.
    - ``cy`` is an updated cell states whose shape is the same as
      ``cx``.
    - ``ys`` is a list of :class:`~chainerx.array` . Each element
      ``ys[t]`` holds hidden states of the last layer corresponding
      to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is
      the mini-batch size for time ``t``, and ``N`` is size of hidden
      units. Note that ``B_t`` is the same value as ``xs[t]``.

.. note::
   The dimension of hidden units is limited to only one size ``N``. If you
   want to use variable dimension of hidden units, please use
   :class:`chainerx.lstm`.

.. seealso::
   :func:`chainerx.lstm`

.. admonition:: Example

    >>> import chainerx as chx
    >>> batchs = [3, 2, 1]  # support variable length sequences
    >>> in_size, out_size, n_layers = 3, 2, 2
    >>> xs = [chx.ones((b, in_size)).astype(chx.float32) for b in batchs]
    >>> [x.shape for x in xs]
    [(3, 3), (2, 3), (1, 3)]
    >>> h_shape = (n_layers, batchs[0], out_size)
    >>> hx = chx.ones(h_shape).astype(chx.float32)
    >>> cx = chx.ones(h_shape).astype(chx.float32)
    >>> w_in = lambda i, j: in_size if i == 0 and j < 4 else out_size
    >>> ws = []
    >>> bs = []
    >>> for n in range(n_layers):
    ...     ws.append([chx.ones((out_size, w_in(n, i))).\
astype(np.float32) for i in range(8)])
    ...     bs.append([chx.ones((out_size,)).astype(chx.float32) \
for _ in range(8)])
    ...
    >>> ws[0][0].shape  # ws[0][:4].shape are (out_size, in_size)
    (2, 3)
    >>> ws[1][0].shape  # others are (out_size, out_size)
    (2, 2)
    >>> bs[0][0].shape
    (2,)
    >>> hy, cy, ys = chx.n_step_lstm(
    ...     n_layers, hx, cx, ws, bs, xs)
    >>> hy.shape
    (2, 3, 2)
    >>> cy.shape
    (2, 3, 2)
    >>> [y.shape for y in ys]
    [(3, 2), (2, 2), (1, 2)]
""")

    _docs.set_doc(
        chainerx.n_step_bilstm,
        """n_step_bilstm(n_layers, hx, cx, ws, bs, xs)
Stacked Bi-directional Long Short-Term Memory function.
This function calculates stacked Bi-directional LSTM with sequences.
This function gets an initial hidden state :math:`h_0`, an initial cell
state :math:`c_0`, an input sequence :math:`x`, weight matrices :math:`W`,
and bias vectors :math:`b`.
This function calculates hidden states :math:`h_t` and :math:`c_t` for each
time :math:`t` from input :math:`x_t`.

.. math::
    i^{f}_t &=& \\sigma(W^{f}_0 x_t + W^{f}_4 h_{t-1} + b^{f}_0 + b^{f}_4),
    \\\\
    f^{f}_t &=& \\sigma(W^{f}_1 x_t + W^{f}_5 h_{t-1} + b^{f}_1 + b^{f}_5),
    \\\\
    o^{f}_t &=& \\sigma(W^{f}_2 x_t + W^{f}_6 h_{t-1} + b^{f}_2 + b^{f}_6),
    \\\\
    a^{f}_t &=& \\tanh(W^{f}_3 x_t + W^{f}_7 h_{t-1} + b^{f}_3 + b^{f}_7),
    \\\\
    c^{f}_t &=& f^{f}_t \\cdot c^{f}_{t-1} + i^{f}_t \\cdot a^{f}_t,
    \\\\
    h^{f}_t &=& o^{f}_t \\cdot \\tanh(c^{f}_t),
    \\\\
    i^{b}_t &=& \\sigma(W^{b}_0 x_t + W^{b}_4 h_{t-1} + b^{b}_0 + b^{b}_4),
    \\\\
    f^{b}_t &=& \\sigma(W^{b}_1 x_t + W^{b}_5 h_{t-1} + b^{b}_1 + b^{b}_5),
    \\\\
    o^{b}_t &=& \\sigma(W^{b}_2 x_t + W^{b}_6 h_{t-1} + b^{b}_2 + b^{b}_6),
    \\\\
    a^{b}_t &=& \\tanh(W^{b}_3 x_t + W^{b}_7 h_{t-1} + b^{b}_3 + b^{b}_7),
    \\\\
    c^{b}_t &=& f^{b}_t \\cdot c^{b}_{t-1} + i^{b}_t \\cdot a^{b}_t, \\\\
    h^{b}_t &=& o^{b}_t \\cdot \\tanh(c^{b}_t), \\\\
    h_t &=& [h^{f}_t; h^{b}_t]

where :math:`W^{f}` is the weight matrices for forward-LSTM, :math:`W^{b}`
is weight matrices for backward-LSTM.
As the function accepts a sequence, it calculates :math:`h_t` for all
:math:`t` with one call. Eight weight matrices and eight bias vectors are
required for each layer of each direction. So, when :math:`S` layers
exist, you need to prepare :math:`16S` weight matrices and :math:`16S`
bias vectors.
If the number of layers ``n_layers`` is greater than :math:`1`, the input
of the ``k``-th layer is the hidden state ``h_t`` of the ``k-1``-th layer.
Note that all input variables except the first layer may have different
shape from the first layer.

Args:
    n_layers(int): The number of layers.
    hx (:class:`~chainerx.array`):
        Variable holding stacked hidden states.
        Its shape is ``(2S, B, N)`` where ``S`` is the number of layers and
        is equal to ``n_layers``, ``B`` is the mini-batch size, and ``N``
        is the dimension of the hidden units. Because of bi-direction, the
        first dimension length is ``2S``.
    cx (:class:`~chainerx.array`): Variable holding stacked cell states.
        It has the same shape as ``hx``.
    ws (list of list of :class:`~chainerx.array`): Weight matrices.
        ``ws[2 * l + m]`` represents the weights for the l-th layer of
        the m-th direction. (``m == 0`` means the forward direction and
        ``m == 1`` means the backward direction.) Each ``ws[i]`` is a
        list containing eight matrices. ``ws[i][j]`` corresponds to
        :math:`W_j` in the equation. ``ws[0][j]`` and ``ws[1][j]`` where
        ``0 <= j < 4`` are ``(N, I)``-shaped because they are multiplied
        with input variables, where ``I`` is the size of the input.
        ``ws[i][j]`` where ``2 <= i`` and ``0 <= j < 4`` are
        ``(N, 2N)``-shaped because they are multiplied with two hidden
        layers :math:`h_t = [h^{f}_t; h^{b}_t]`. All other matrices are
        ``(N, N)``-shaped.
    bs (list of list of :class:`~chainerx.array`): Bias vectors.
        ``bs[2 * l + m]`` represents the weights for the l-th layer of
        m-th direction. (``m == 0`` means the forward direction and
        ``m == 1`` means the backward direction.)
        Each ``bs[i]`` is a list containing eight vectors.
        ``bs[i][j]`` corresponds to :math:`b_j` in the equation.
        The shape of each matrix is ``(N,)``.
    xs (list of :class:`~chainerx.array`):
        A list of :class:`~chainerx.array`
        holding input values. Each element ``xs[t]`` holds input value
        for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is the
        mini-batch size for time ``t``.
        When sequences has different lengths, they must be
        sorted in descending order of their lengths.
        So ``xs`` needs to satisfy
        ``xs[t].shape[0] >= xs[t + 1].shape[0]``.

Returns:
    tuple: This function returns a tuple containing three elements,
    ``hy``, ``cy`` and ``ys``.

    - ``hy`` is an updated hidden states whose shape is the same as
      ``hx``.
    - ``cy`` is an updated cell states whose shape is the same as
      ``cx``.
    - ``ys`` is a list of :class:`~chainer.array` . Each element
      ``ys[t]`` holds hidden states of the last layer corresponding
      to an input ``xs[t]``. Its shape is ``(B_t, 2N)`` where ``B_t``
      is the mini-batch size for time ``t``, and ``N`` is size of
      hidden units. Note that ``B_t`` is the same value as ``xs[t]``.

.. admonition:: Example

    >>> import chainerx as chx
    >>> batchs = [3, 2, 1]  # support variable length sequences
    >>> in_size, out_size, n_layers = 3, 2, 2
    >>> dropout_ratio = 0.0
    >>> xs = [chx.ones((b, in_size)).astype(chx.float32) for b in batchs]
    >>> [x.shape for x in xs]
    [(3, 3), (2, 3), (1, 3)]
    >>> h_shape = (n_layers * 2, batchs[0], out_size)
    >>> hx = chx.ones(h_shape).astype(chx.float32)
    >>> cx = chx.ones(h_shape).astype(chx.float32)
    >>> def w_in(i, j):
    ...     if i == 0 and j < 4:
    ...         return in_size
    ...     elif i > 0 and j < 4:
    ...         return out_size * 2
    ...     else:
    ...         return out_size
    ...
    >>> ws = []
    >>> bs = []
    >>> for n in range(n_layers):
    ...     for direction in (0, 1):
    ...         ws.append([chx.ones((out_size, w_in(n, i))).\
astype(np.float32) for i in range(8)])
    ...         bs.append([chx.ones((out_size,)).astype(chx.float32) \
for _ in range(8)])
    ...
    >>> ws[0][0].shape  # ws[0:2][:4].shape are (out_size, in_size)
    (2, 3)
    >>> ws[2][0].shape  # ws[2:][:4].shape are (out_size, 2 * out_size)
    (2, 4)
    >>> ws[0][4].shape  # others are (out_size, out_size)
    (2, 2)
    >>> bs[0][0].shape
    (2,)
    >>> hy, cy, ys = chx.n_step_bilstm(
    ...     n_layers, hx, cx, ws, bs, xs)
    >>> hy.shape
    (4, 3, 2)
    >>> cy.shape
    (4, 3, 2)
    >>> [y.shape for y in ys]
    [(3, 4), (2, 4), (1, 4)]
    """)

    _docs.set_doc(
        chainerx.n_step_gru,
        """n_step_gru(n_layers, hx, ws, bs, xs)
Stacked Uni-directional Gated Recurrent Unit function.
This function calculates stacked Uni-directional GRU with sequences.
This function gets an initial hidden state :math:`h_0`, an input
sequence :math:`x`, weight matrices :math:`W`, and bias vectors :math:`b`.
This function calculates hidden states :math:`h_t` for each time :math:`t`
from input :math:`x_t`.

.. math::
   r_t &= \\sigma(W_0 x_t + W_3 h_{t-1} + b_0 + b_3) \\\\
   z_t &= \\sigma(W_1 x_t + W_4 h_{t-1} + b_1 + b_4) \\\\
   h'_t &= \\tanh(W_2 x_t + b_2 + r_t \\cdot (W_5 h_{t-1} + b_5)) \\\\
   h_t &= (1 - z_t) \\cdot h'_t + z_t \\cdot h_{t-1}

As the function accepts a sequence, it calculates :math:`h_t` for all
:math:`t` with one call. Six weight matrices and six bias vectors are
required for each layers. So, when :math:`S` layers exists, you need to
prepare :math:`6S` weight matrices and :math:`6S` bias vectors.
If the number of layers ``n_layers`` is greather than :math:`1`, input
of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.
Note that all input variables except first layer may have different shape
from the first layer.

Args:
    n_layers(int): Number of layers.
    hx (~chainerx.array):
        Variable holding stacked hidden states.
        Its shape is ``(S, B, N)`` where ``S`` is number of layers and is
        equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
        dimension of hidden units.
    ws (list of list of :class:`~chainerx.array`): Weight matrices.
        ``ws[i]`` represents weights for i-th layer.
        Each ``ws[i]`` is a list containing six matrices.
        ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
        Only ``ws[0][j]`` where ``0 <= j < 3`` is ``(N, I)`` shape as they
        are multiplied with input variables. All other matrices has
        ``(N, N)`` shape.
    bs (list of list of :class:`~chainerx.array`): Bias vectors.
        ``bs[i]`` represnents biases for i-th layer.
        Each ``bs[i]`` is a list containing six vectors.
        ``bs[i][j]`` is corresponding with ``b_j`` in the equation.
        Shape of each matrix is ``(N,)`` where ``N`` is dimension of
        hidden units.
    xs (list of :class:`~chainerx.array`):
        A list of :class:`~chainerx.array`
        holding input values. Each element ``xs[t]`` holds input value
        for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is
        mini-batch size for time ``t``, and ``I`` is size of input units.
        Note that this function supports variable length sequences.
        When sequneces has different lengths, sort sequences in descending
        order by length.
        So ``xs`` needs to satisfy
        ``xs[t].shape[0] >= xs[t + 1].shape[0]``.

Returns:
    tuple: This function returns a tuple containing two elements,
    ``hy`` and ``ys``.

    - ``hy`` is an updated hidden states whose shape is same as ``hx``.
    - ``ys`` is a list of :class:`~chainerx.array` . Each element
      ``ys[t]`` holds hidden states of the last layer corresponding
      to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is
      mini-batch size for time ``t``, and ``N`` is size of hidden
      units. Note that ``B_t`` is the same value as ``xs[t]``
    """)

    _docs.set_doc(
        chainerx.n_step_bigru,
        """n_step_bigru(n_layers, hx, ws, bs, xs)
Stacked Bi-directional Gated Recurrent Unit function.
This function calculates stacked Bi-directional GRU with sequences.
This function gets an initial hidden state :math:`h_0`, an input
sequence :math:`x`, weight matrices :math:`W`, and bias vectors :math:`b`.
This function calculates hidden states :math:`h_t` for each time :math:`t`
from input :math:`x_t`.

.. math::
   r^{f}_t &= \\sigma(W^{f}_0 x_t + W^{f}_3 h_{t-1} + b^{f}_0 + b^{f}_3)
   \\\\
   z^{f}_t &= \\sigma(W^{f}_1 x_t + W^{f}_4 h_{t-1} + b^{f}_1 + b^{f}_4)
   \\\\
   h^{f'}_t &= \\tanh(W^{f}_2 x_t + b^{f}_2 + r^{f}_t \\cdot (W^{f}_5
   h_{t-1} + b^{f}_5)) \\\\
   h^{f}_t &= (1 - z^{f}_t) \\cdot h^{f'}_t + z^{f}_t \\cdot h_{t-1}
   \\\\
   r^{b}_t &= \\sigma(W^{b}_0 x_t + W^{b}_3 h_{t-1} + b^{b}_0 + b^{b}_3)
   \\\\
   z^{b}_t &= \\sigma(W^{b}_1 x_t + W^{b}_4 h_{t-1} + b^{b}_1 + b^{b}_4)
   \\\\
   h^{b'}_t &= \\tanh(W^{b}_2 x_t + b^{b}_2 + r^{b}_t \\cdot (W^{b}_5
   h_{t-1} + b^{b}_5)) \\\\
   h^{b}_t &= (1 - z^{b}_t) \\cdot h^{b'}_t + z^{b}_t \\cdot h_{t-1}
   \\\\
   h_t  &= [h^{f}_t; h^{b}_t] \\\\

where :math:`W^{f}` is weight matrices for forward-GRU, :math:`W^{b}` is
weight matrices for backward-GRU.
As the function accepts a sequence, it calculates :math:`h_t` for all
:math:`t` with one call. Six weight matrices and six bias vectors are
required for each layers. So, when :math:`S` layers exists, you need to
prepare :math:`6S` weight matrices and :math:`6S` bias vectors.
If the number of layers ``n_layers`` is greather than :math:`1`, input
of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.
Note that all input variables except first layer may have different shape
from the first layer.

Args:
    n_layers(int): Number of layers.
    hx (:class:`~chainerx.array`):
        Variable holding stacked hidden states.
        Its shape is ``(2S, B, N)`` where ``S`` is number of layers and is
        equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
        dimension of hidden units.
    ws (list of list of :class:`~chainerx.array`): Weight matrices.
        ``ws[i]`` represents weights for i-th layer.
        Each ``ws[i]`` is a list containing six matrices.
        ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
        Only ``ws[0][j]`` where ``0 <= j < 3`` is ``(N, I)`` shape as they
        are multiplied with input variables. All other matrices has
        ``(N, N)`` shape.
    bs (list of list of :class:`~chainerx.array`): Bias vectors.
        ``bs[i]`` represnents biases for i-th layer.
        Each ``bs[i]`` is a list containing six vectors.
        ``bs[i][j]`` is corresponding with ``b_j`` in the equation.
        Shape of each matrix is ``(N,)`` where ``N`` is dimension of
        hidden units.
    xs (list of :class:`~chainerx.array`):
        A list of :class:`~chainerx.array` holding input values.
        Each element ``xs[t]`` holds input value
        for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is
        mini-batch size for time ``t``, and ``I`` is size of input units.
        Note that this function supports variable length sequences.
        When sequneces has different lengths, sort sequences in descending
        order by length.
        So ``xs`` needs to satisfy
        ``xs[t].shape[0] >= xs[t + 1].shape[0]``.

Returns:
    tuple: This function returns a tuple containing two elements,
    ``hy`` and ``ys``.

    - ``hy`` is an updated hidden states whose shape is same as ``hx``.
    - ``ys`` is a list of :class:`~chainerx.array` . Each element
      ``ys[t]`` holds hidden states of the last layer corresponding
      to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is
      mini-batch size for time ``t``, and ``N`` is size of hidden
      units. Note that ``B_t`` is the same value as ``xs[t]``.
    """)

    _docs.set_doc(
        chainerx.n_step_rnn,
        """n_step_rnn(n_layers, hx, ws, bs, xs, activation='tanh')
Stacked Uni-directional RNN function for sequence inputs.
This function calculates stacked Uni-directional RNN with sequences.
This function gets an initial hidden state :math:`h_0`,
an initial cell state :math:`c_0`, an input sequence :math:`x`,
weight matrices :math:`W`, and bias vectors :math:`b`.
This function calculates hidden states :math:`h_t` and :math:`c_t` for each
time :math:`t` from input :math:`x_t`.

.. math::
   h_t = f(W_0 x_t + W_1 h_{t-1} + b_0 + b_1)

where :math:`f` is an activation function.
Weight matrices :math:`W` contains two matrices :math:`W_0` and
:math:`W_1`. :math:`W_0` is a parameter for an input sequence.
:math:`W_1` is a parameter for a hidden state.
Bias matrices :math:`b` contains two matrices :math:`b_0` and :math:`b_1`.
:math:`b_0` is a parameter for an input sequence.
:math:`b_1` is a parameter for a hidden state.
As the function accepts a sequence, it calculates :math:`h_t` for all
:math:`t` with one call. Two weight matrices and two bias vectors are
required for each layer. So, when :math:`S` layers exist, you need to
prepare :math:`2S` weight matrices and :math:`2S` bias vectors.
If the number of layers ``n_layers`` is greather than :math:`1`, input
of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.
Note that all input variables except first layer may have different shape
from the first layer.

Args:
    n_layers(int): Number of layers.
    hx (:class:`~chainerx.array`):
        Variable holding stacked hidden states.
        Its shape is ``(S, B, N)`` where ``S`` is number of layers and is
        equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
        dimension of hidden units.
    ws (list of list of :class:`~chainerx.array`): Weight matrices.
        ``ws[i]`` represents weights for i-th layer.
        Each ``ws[i]`` is a list containing two matrices.
        ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
        Only ``ws[0][j]`` where ``0 <= j < 1`` is ``(N, I)`` shape as they
        are multiplied with input variables. All other matrices has
        ``(N, N)`` shape.
    bs (list of list of :class:`~chainerx.array`): Bias vectors.
        ``bs[i]`` represnents biases for i-th layer.
        Each ``bs[i]`` is a list containing two vectors.
        ``bs[i][j]`` is corresponding with ``b_j`` in the equation.
        Shape of each matrix is ``(N,)`` where ``N`` is dimension of
        hidden units.
    xs (list of :class:`~chainerx.array`):
        A list of :class:`~chainerx.array` holding input values.
        Each element ``xs[t]`` holds input value for time ``t``.
        Its shape is ``(B_t, I)``, where ``B_t`` is
        mini-batch size for time ``t``, and ``I`` is size of input units.
        Note that this function supports variable length sequences.
        When sequneces has different lengths, sort sequences in descending
        order by length.
        So ``xs`` needs to satisfy
        ``xs[t].shape[0] >= xs[t + 1].shape[0]``.
    activation (str): Activation function name.
        Please select ``tanh`` or ``relu``.

Returns:
    tuple: This function returns a tuple containing two elements,
    ``hy`` and ``ys``.

    - ``hy`` is an updated hidden states whose shape is same as ``hx``.
    - ``ys`` is a list of :class:`~chainerx.array` . Each element
      ``ys[t]`` holds hidden states of the last layer corresponding
      to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is
      mini-batch size for time ``t``, and ``N`` is size of hidden
      units. Note that ``B_t`` is the same value as ``xs[t]``.
    """)

    _docs.set_doc(
        chainerx.n_step_birnn,
        """n_step_birnn(n_layers, hx, ws, bs, xs, activation='tanh')
Stacked Bi-directional RNN function for sequence inputs.
This function calculates stacked Bi-directional RNN with sequences.
This function gets an initial hidden state :math:`h_0`, an initial
cell state :math:`c_0`, an input sequence :math:`x`,
weight matrices :math:`W`, and bias vectors :math:`b`.
This function calculates hidden states :math:`h_t` and :math:`c_t` for each
time :math:`t` from input :math:`x_t`.

.. math::
    h^{f}_t &=& f(W^{f}_0 x_t + W^{f}_1 h_{t-1} + b^{f}_0 + b^{f}_1), \\\\
    h^{b}_t &=& f(W^{b}_0 x_t + W^{b}_1 h_{t-1} + b^{b}_0 + b^{b}_1), \\\\
    h_t  &=& [h^{f}_t; h^{f}_t], \\\\

where :math:`f` is an activation function.
Weight matrices :math:`W` contains two matrices :math:`W^{f}` and
:math:`W^{b}`. :math:`W^{f}` is weight matrices for forward directional
RNN. :math:`W^{b}` is weight matrices for backward directional RNN.
:math:`W^{f}` contains :math:`W^{f}_0` for an input sequence and
:math:`W^{f}_1` for a hidden state.
:math:`W^{b}` contains :math:`W^{b}_0` for an input sequence and
:math:`W^{b}_1` for a hidden state.
Bias matrices :math:`b` contains two matrices :math:`b^{f}` and
:math:`b^{f}`. :math:`b^{f}` contains :math:`b^{f}_0` for an input sequence
and :math:`b^{f}_1` for a hidden state.
:math:`b^{b}` contains :math:`b^{b}_0` for an input sequence and
:math:`b^{b}_1` for a hidden state.
As the function accepts a sequence, it calculates :math:`h_t` for all
:math:`t` with one call. Two weight matrices and two bias vectors are
required for each layer. So, when :math:`S` layers exist, you need to
prepare :math:`2S` weight matrices and :math:`2S` bias vectors.
If the number of layers ``n_layers`` is greather than :math:`1`, input
of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.
Note that all input variables except first layer may have different shape
from the first layer.

Args:
    n_layers(int): Number of layers.
    hx (:class:`~chainerx.array`):
        Variable holding stacked hidden states.
        Its shape is ``(2S, B, N)`` where ``S`` is number of layers and is
        equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
        dimension of hidden units. Because of bi-direction, the
        first dimension length is ``2S``.
    ws (list of list of :class:`~chainerx.array`): Weight matrices.
        ``ws[i + di]`` represents weights for i-th layer.
        Note that ``di = 0`` for forward-RNN and ``di = 1`` for
        backward-RNN.
        Each ``ws[i + di]`` is a list containing two matrices.
        ``ws[i + di][j]`` is corresponding with ``W^{f}_j`` if ``di = 0``
        and corresponding with ``W^{b}_j`` if ``di = 1`` in the equation.
        Only ``ws[0][j]`` and ``ws[1][j]`` where ``0 <= j < 1`` are
        ``(I, N)`` shape as they are multiplied with input variables.
        All other matrices has ``(N, N)`` shape.
    bs (list of list of :class:`~chainerx.array`): Bias vectors.
        ``bs[i + di]`` represnents biases for i-th layer.
        Note that ``di = 0`` for forward-RNN and ``di = 1`` for
        backward-RNN.
        Each ``bs[i + di]`` is a list containing two vectors.
        ``bs[i + di][j]`` is corresponding with ``b^{f}_j`` if ``di = 0``
        and corresponding with ``b^{b}_j`` if ``di = 1`` in the equation.
        Shape of each matrix is ``(N,)`` where ``N`` is dimension of
        hidden units.
    xs (list of :class:`~chainerx.array`):
        A list of :class:`~chainerx.array` holding input values.
        Each element ``xs[t]`` holds input value
        for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is
        mini-batch size for time ``t``, and ``I`` is size of input units.
        Note that this function supports variable length sequences.
        When sequneces has different lengths, sort sequences in descending
        order by length.
        So ``xs`` needs to satisfy
        ``xs[t].shape[0] >= xs[t + 1].shape[0]``.
    activation (str): Activation function name.
        Please select ``tanh`` or ``relu``.

Returns:
    tuple: This function returns a tuple containing two elements,
    ``hy`` and ``ys``.

    - ``hy`` is an updated hidden states whose shape is same as ``hx``.
    - ``ys`` is a list of :class:`~chainerx.array` . Each element
      ``ys[t]`` holds hidden states of the last layer corresponding
      to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t``
      is mini-batch size for time ``t``, and ``N`` is size of hidden
      units. Note that ``B_t`` is the same value as ``xs[t]``.
    """)
