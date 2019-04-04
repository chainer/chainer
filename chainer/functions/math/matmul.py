import warnings

import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer import utils
from chainer.utils import type_check
import chainerx


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
    xp = backend.get_array_module(a)

    if hasattr(xp, 'matmul'):  # numpy.matmul is supported from version 1.10.0
        return xp.matmul(a, b)
    if a.ndim <= 2 or b.ndim <= 2:
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
        type_check._argname(in_types, ('a', 'b'))
        a_type, b_type = in_types

        type_check.expect(
            a_type.dtype.kind == 'f',
            b_type.dtype.kind == 'f',
            a_type.ndim >= 1,
            b_type.ndim >= 1,
        )

        a_ndim = type_check.eval(a_type.ndim)
        b_ndim = type_check.eval(b_type.ndim)
        if b_ndim == 1:
            a_idx = -2 if self.transa and a_ndim > 1 else -1
            type_check.expect(a_type.shape[a_idx] == b_type.shape[0])
        elif a_ndim == 1:
            b_idx = -1 if self.transb and b_ndim > 1 else -2
            type_check.expect(a_type.shape[0] == b_type.shape[b_idx])
        else:
            a_idx = _get_check_index(self.transa, False,
                                     row_idx=-2, col_idx=-1)
            b_idx = _get_check_index(self.transb, True,
                                     row_idx=-2, col_idx=-1)
            type_check.expect(a_type.shape[a_idx] == b_type.shape[b_idx])
            type_check.expect_broadcast_shapes(
                a_type.shape[:-2], b_type.shape[:-2])

    def forward_chainerx(self, x):
        a, b = x
        # TODO(sonots): Support transa and transb in ChainerX
        if self.transa or self.transb or self.transc:
            return chainer.Fallback
        # TODO(sonots): Support dtype promotion in ChainerX
        if a.dtype != b.dtype:
            return chainer.Fallback
        # TODO(sonots): Support ndim > 2 in ChainerX
        if a.ndim != 2 or b.ndim != 2:
            return chainer.Fallback
        # TODO(niboshi): Support it
        if self.dtype is not None and self.dtype != a.dtype:
            return chainer.Fallback
        return chainerx.dot(a, b),

    def forward(self, x):
        self.retain_inputs((0, 1))
        a, b = x
        # may broadcast
        y = _matmul(a, b, self.transa, self.transb, self.transc)

        if self.dtype is not None:
            dtype = self.dtype
        else:
            dtype = y.dtype
        return utils.force_array(y, dtype),

    def backward(self, indexes, grad_outputs):
        a, b = self.get_retained_inputs()
        gy, = grad_outputs
        is_a_vector = a.ndim == 1
        is_b_vector = b.ndim == 1

        ret = []
        if 0 in indexes:
            if is_b_vector:
                u, v = chainer.functions.cast(gy, b.dtype), b
                if not is_a_vector:
                    if self.transa:
                        u, v = v, u
                    u = chainer.functions.expand_dims(u, -1)
                    v = (chainer.functions.expand_dims(v, -2) if v.ndim > 1
                         else v)
                ga = chainer.functions.cast(u * v, a.dtype)
            elif is_a_vector:
                bt = chainer.functions.rollaxis(b, -1 if self.transb else -2)
                ga = chainer.functions.tensordot(bt, gy, axes=gy.ndim)
                ga = chainer.functions.cast(ga, a.dtype)
            else:
                ga, = MatMul(self.transc, not self.transb, self.transa,
                             a.dtype).apply((gy, b))
                ga = chainer.functions.sum_to(ga, a.shape)
            ret.append(ga)

        if 1 in indexes:
            if is_a_vector:
                u, v = a, chainer.functions.cast(gy, a.dtype)
                if not is_b_vector:
                    if self.transb:
                        u, v = v, u
                    u = chainer.functions.expand_dims(u, -1)
                    v = (chainer.functions.expand_dims(v, -2) if v.ndim > 1
                         else v)
                gb = chainer.functions.cast(u * v, b.dtype)
            elif is_b_vector:
                at = chainer.functions.rollaxis(a, -2 if self.transa else -1)
                gb = chainer.functions.tensordot(at, gy, axes=gy.ndim)
                gb = chainer.functions.cast(gb, b.dtype)
            else:
                gb, = MatMul(not self.transa, self.transc, self.transb,
                             b.dtype).apply((a, gy))
                gb = chainer.functions.sum_to(gb, b.shape)
            ret.append(gb)

        return ret


def matmul(a, b, transa=False, transb=False):
    """Computes the matrix multiplication of two arrays.

    Args:
        a (:class:`~chainer.Variable` or :ref:`ndarray`):
            The left operand of the matrix multiplication.
            If ``a`` and ``b`` are both 1-D arrays, ``matmul`` returns a dot
            product of vector `a` and vector `b`. If 2-D arrays, ``matmul``
            returns matrix product of ``a`` and ``b``. If either's dimension is
            larger than 2, they are treated as a stack of matrices residing in
            the last two indexes. ``matmul`` returns a stack of each two
            arrays. In this case, ``a`` and ``b`` are broadcasted along axes
            except the last two.
        b (:class:`~chainer.Variable` or :ref:`ndarray`):
            The right operand of the matrix multiplication.
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
        >>> F.matmul(a, b).array
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
        type_check._argname(in_types, ('a', 'b'))
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
        a (:class:`~chainer.Variable` or :ref:`ndarray`):
            The left operand of the batch matrix multiplications.
            A 2-D array of shape ``(B, N)`` is considered as B
            :math:`N \\times 1` matrices.
            A 3-D array of shape ``(B, M, N)`` is considered as B
            :math:`M \\times N` matrices.
        b (:class:`~chainer.Variable` or :ref:`ndarray`):
            The right operand of the batch matrix multiplications.
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
