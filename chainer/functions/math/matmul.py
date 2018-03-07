import warnings

import numpy

from chainer.backends import cuda
from chainer import function_node
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

    if hasattr(xp, 'matmul'):  # numpy.matmul is supported from version 1.10.0
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


class MatMul(function_node.FunctionNode):

    def __init__(self, transa=False, transb=False, transc=False, dtype=None):
        self.transa = transa
        self.transb = transb
        self.transc = transc
        self.dtype = dtype

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        a_type, b_type = in_types

        type_check.expect(
            a_type.dtype.kind == 'f',
            b_type.dtype.kind == 'f',
        )

        a_ndim = type_check.eval(a_type.ndim)
        b_ndim = type_check.eval(b_type.ndim)
        if a_ndim == 0 or b_ndim == 0:
            pass
        elif a_ndim == 1 or b_ndim == 1:
            type_check.expect(
                a_type.ndim == b_type.ndim,
                a_type.shape == b_type.shape,
            )
        else:
            a_idx = _get_check_index(self.transa, False,
                                     row_idx=-2, col_idx=-1)
            b_idx = _get_check_index(self.transb, True,
                                     row_idx=-2, col_idx=-1)
            type_check.expect(
                a_type.ndim == b_type.ndim,
                a_type.shape[:-2] == b_type.shape[:-2],
                a_type.shape[a_idx] == b_type.shape[b_idx],
            )

    def forward(self, x):
        self.retain_inputs((0, 1))
        a, b = x
        if a.ndim == 0 or b.ndim == 0:
            y = a * b
        else:
            y = _matmul(a, b, self.transa, self.transb, self.transc)
        if self.dtype is not None and y.dtype != self.dtype:
            y = y.astype(self.dtype)
        return utils.force_array(y),

    def backward(self, indexes, grad_outputs):
        a, b = self.get_retained_inputs()
        gy, = grad_outputs

        ga = None
        if 0 in indexes:
            ga, = MatMul(self.transc, not self.transb, self.transa,
                         a.dtype).apply((gy, b))

        gb = None
        if 1 in indexes:
            gb, = MatMul(not self.transa, self.transc, self.transb,
                         b.dtype).apply((a, gy))

        return ga, gb


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

        >>> a = np.array([[1, 0], [0, 1]], np.float32)
        >>> b = np.array([[4, 1], [2, 2]], np.float32)
        >>> F.matmul(a, b).data
        array([[4., 1.],
               [2., 2.]], dtype=float32)

    """
    return MatMul(transa=transa, transb=transb).apply((a, b))[0]


def _get_size(typ, index):
    if index == 2 and type_check.eval(typ.ndim) == 2:
        return 1
    else:
        return typ.shape[index]


def _batch_matmul(a, b, transa, transb, transout):
    a = a.reshape(a.shape[:2] + (-1,))
    b = b.reshape(b.shape[:2] + (-1,))
    return _matmul(a, b, transa, transb, transout)


class BatchMatMul(function_node.FunctionNode):

    def __init__(self, transa=False, transb=False):
        self.transa = transa
        self.transb = transb

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        a_type, b_type = in_types

        type_check.expect(
            a_type.dtype == numpy.float32,
            b_type.dtype == numpy.float32
        )

        _check_ndim(a_type, lower=2, upper=3)
        _check_ndim(b_type, lower=2, upper=3)

        a_idx = _get_check_index(self.transa, False, row_idx=1, col_idx=2)
        b_idx = _get_check_index(self.transb, True, row_idx=1, col_idx=2)
        a_size = _get_size(a_type, a_idx)
        b_size = _get_size(b_type, b_idx)
        type_check.expect(
            a_size == b_size
        )

    def forward(self, x):
        self.retain_inputs((0, 1))
        a, b = x
        return _batch_matmul(a, b, self.transa, self.transb, False),

    def backward(self, indexes, grad_outputs):
        a, b = self.get_retained_inputs()
        return BatchMatMulGrad(self.transa, self.transb).apply(
            (a, b, grad_outputs[0]))


class BatchMatMulGrad(function_node.FunctionNode):

    def __init__(self, transa=False, transb=False):
        self.transa = transa
        self.transb = transb

    def forward(self, inputs):
        self.retain_inputs((0, 1, 2))
        a, b, gy = inputs
        ga = _batch_matmul(gy, b, False, not self.transb,
                           self.transa).reshape(a.shape)
        gb = _batch_matmul(a, gy, not self.transa, False,
                           self.transb).reshape(b.shape)
        return ga, gb

    def backward(self, indexes, grad_outputs):
        a, b, gy = self.get_retained_inputs()
        gga, ggb = grad_outputs

        ret = []
        if 0 in indexes or 1 in indexes:
            ga, gb = BatchMatMulGrad(self.transa, self.transb).apply(
                (gga, ggb, gy))
            if 0 in indexes:
                ret.append(ga)
            if 1 in indexes:
                ret.append(gb)
        if 2 in indexes:
            ret.append(
                BatchMatMul(self.transa, self.transb).apply((gga, b))[0] +
                BatchMatMul(self.transa, self.transb).apply((a, ggb))[0])
        return ret


def batch_matmul(a, b, transa=False, transb=False):
    """Computes the batch matrix multiplications of two sets of arrays.

    Args:
        a (Variable): The left operand of the batch matrix multiplications.
            A 2-D array of shape ``(B, N)`` is considered as B
            :math:`N \\times 1` matrices.
            A 3-D array of shape ``(B, M, N)`` is considered as B
            :math:`M \\times N` matrices.
        b (Variable): The right operand of the batch matrix multiplications.
            Its array is treated as matrices in the same way as ``a``'s array.
        transa (bool): If ``True``, transpose each matrix in ``a``.
        transb (bool): If ``True``, transpose each matrix in ``b``.

    Returns:
        ~chainer.Variable: The result of the batch matrix multiplications as a
        3-D array.

    .. deprecated:: v3.0.0
       batch_matmul is deprecated. Use ``matmul`` instead.

    """
    warnings.warn('batch_matmul is deprecated. Use matmul instead.',
                  DeprecationWarning)
    return BatchMatMul(transa=transa, transb=transb).apply((a, b))[0]
