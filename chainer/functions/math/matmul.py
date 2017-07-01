import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


def _mat_ptrs(a):
    """Creates an array of pointers to matrices

    Args:
        a: A batch of matrices on GPU.
    Returns:
        GPU array of pointers to matrices.
    """
    if len(a) == 1:
        return cuda.cupy.full((1,), a.data.ptr, dtype=numpy.uintp)
    else:
        stride = a.strides[0]
        ptr = a.data.ptr
        return cuda.cupy.arange(ptr, ptr + stride * len(a), stride,
                                dtype=numpy.uintp)


def _as_batch_mat(x):
    return x.reshape(len(x), x.shape[1], -1)


def _get_ld(a):
    strides = a.strides[-2:]
    trans = numpy.argmin(strides)
    return trans, int(max(a.shape[trans - 2], max(strides) // a.itemsize))


def _matmul(a, b, transa=False, transb=False, transout=False):
    if transout:
        transa, transb = not transb, not transa
        a, b = b, a
    if transa and a.ndim != 1:
        a = a.swapaxes(-1, -2)
    if transb and b.ndim != 1:
        b = b.swapaxes(-1, -2)
    xp = cuda.get_array_module(a)

    if hasattr(xp, 'matmul'):
        return xp.matmul(a, b)
    if a.ndim <= 2:
        return numpy.dot(a, b)
    else:
        return numpy.einsum('...ij,...jk->...ik', a, b)


def _check_ndim(in_type, lower=1, upper=2):
    type_check.expect(
        in_type.ndim >= lower,
        in_type.ndim <= upper
    )


def _get_check_index(trans, right, row_idx=0, col_idx=1):
    if trans ^ right:
        return row_idx
    else:
        return col_idx


class MatMul(function.Function):

    def __init__(self, transa=False, transb=False):
        self.transa = transa
        self.transb = transb

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        a_type, b_type = in_types

        type_check.expect(
            a_type.dtype.kind == 'f',
            b_type.dtype.kind == 'f',
            a_type.ndim >= 1,
            a_type.ndim == b_type.ndim,
        )

        ndim = type_check.eval(a_type.ndim)
        if ndim == 1:
            type_check.expect(a_type.shape == b_type.shape)
        else:
            a_idx = _get_check_index(self.transa, False,
                                     row_idx=-2, col_idx=-1)
            b_idx = _get_check_index(self.transb, True,
                                     row_idx=-2, col_idx=-1)
            type_check.expect(
                a_type.shape[:-2] == b_type.shape[:-2],
                a_type.shape[a_idx] == b_type.shape[b_idx],
            )

    def forward(self, x):
        a, b = x
        y = _matmul(a, b, self.transa, self.transb)
        return utils.force_array(y),

    def backward(self, x, gy):
        a, b = x
        a_shape = a.shape
        b_shape = b.shape

        if gy[0].ndim == 0:
            ga = gy[0] * b
        else:
            ga = _matmul(gy[0], b, False, not self.transb)
        if self.transa and a.ndim != 1:
            ga = ga.swapaxes(-1, -2)
        ga = ga.reshape(a_shape)

        if gy[0].ndim == 0:
            gb = a * gy[0]
        else:
            gb = _matmul(a, gy[0], not self.transa, False)
        if self.transb and a.ndim != 1:
            gb = gb.swapaxes(-1, -2)
        gb = gb.reshape(b_shape)
        return ga.astype(a.dtype), gb.astype(b.dtype)


def matmul(a, b, transa=False, transb=False):
    """Computes the matrix multiplication of two arrays.

    Args:
        a (Variable): The left operand of the matrix multiplication.
            If ``a`` and ``b`` are both 1-D arrays, ``matmul`` returns a dot
            product of vector `a` and vector `b`. If 2-D arrays, ``matmul``
            returns matrix product of ``a`` and ``b``. If arrays' dimension is
            larger than 2, they are treated as a stack of matrices residing in
            the last two indexes. ``matmul`` returns a stack of each two
            arrays. ``a`` and ``b`` must have the same dimension.
        b (Variable): The right operand of the matrix multiplication.
            Its array is treated as a matrix in the same way as ``a``'s array.
        transa (bool): If ``True``, each matrices in ``a`` will be transposed.
            If ``a.ndim == 1``, do nothing.
        transb (bool): If ``True``, each matrices in ``b`` will be transposed.
            If ``b.ndim == 1``, do nothing.

    Returns:
        ~chainer.Variable: The result of the matrix multiplication.

    .. admonition:: Example

        >>> a = np.array([[1, 0], [0, 1]], 'f')
        >>> b = np.array([[4, 1], [2, 2]], 'f')
        >>> F.matmul(a, b).data
        array([[ 4.,  1.],
               [ 2.,  2.]], dtype=float32)

    """
    return MatMul(transa=transa, transb=transb)(a, b)
