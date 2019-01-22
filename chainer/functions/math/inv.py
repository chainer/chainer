import numpy.linalg

import chainer
from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer.functions.math import matmul
from chainer import utils
from chainer.utils import precision
from chainer.utils import type_check


def _inv_gpu(b):
    # We do a batched LU decomposition on the GPU to compute the inverse
    # Change the shape of the array to be size=1 minibatch if necessary
    # Also copy the matrix as the elements will be modified in-place
    a = matmul._as_batch_mat(b).copy()
    n = a.shape[1]
    n_matrices = len(a)
    # Pivot array
    p = cuda.cupy.empty((n, n_matrices), dtype=numpy.int32)
    # Output array
    c = cuda.cupy.empty_like(a)
    # These arrays hold information on the execution success
    # or if the matrix was singular
    info = cuda.cupy.empty(n_matrices, dtype=numpy.int32)
    ap = matmul._mat_ptrs(a)
    cp = matmul._mat_ptrs(c)
    _, lda = matmul._get_ld(a)
    _, ldc = matmul._get_ld(c)
    handle = cuda.Device().cublas_handle
    if b.dtype == numpy.float32:
        cuda.cublas.sgetrfBatched(
            handle, n, ap.data.ptr, lda, p.data.ptr, info.data.ptr, n_matrices)
        cuda.cublas.sgetriBatched(
            handle, n, ap.data.ptr, lda, p.data.ptr, cp.data.ptr, ldc,
            info.data.ptr, n_matrices)
    elif b.dtype == numpy.float64:
        cuda.cublas.dgetrfBatched(
            handle, n, ap.data.ptr, lda, p.data.ptr, info.data.ptr, n_matrices)
        cuda.cublas.dgetriBatched(
            handle, n, ap.data.ptr, lda, p.data.ptr, cp.data.ptr, ldc,
            info.data.ptr, n_matrices)
    else:
        assert False
    return c, info


class Inv(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('a',))
        a_type, = in_types
        type_check.expect(a_type.dtype.kind == 'f')
        # Only 2D array shapes allowed
        type_check.expect(a_type.ndim == 2)
        # Matrix inversion only allowed for square matrices
        type_check.expect(a_type.shape[0] == a_type.shape[1])

    @precision._fp16_mixed_precision_helper
    def forward_cpu(self, x):
        self.retain_outputs((0,))
        try:
            invx = utils.force_array(numpy.linalg.inv(x[0]))
        except numpy.linalg.LinAlgError:
            raise ValueError('Input has singular matrices.')
        return invx,

    @precision._fp16_mixed_precision_helper
    def forward_gpu(self, x):
        self.retain_outputs((0,))
        shape = x[0].shape
        invx, info = _inv_gpu(x[0].reshape(1, *shape))
        if chainer.is_debug():
            if cuda.cupy.any(info != 0):
                raise ValueError('Input has singular matrices.')
        invx = invx.reshape(shape)
        return invx,

    def backward(self, x, gy):
        invx, = self.get_retained_outputs()
        # Gradient is - x^-T (dx) x^-T
        invxT = chainer.functions.transpose(invx)
        gx = chainer.functions.matmul(
            chainer.functions.matmul(- invxT, gy[0]), invxT)
        return gx,


class BatchInv(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('a',))
        a_type, = in_types
        type_check.expect(a_type.dtype.kind == 'f')
        # Only a minibatch of 2D array shapes allowed
        type_check.expect(a_type.ndim == 3)
        # Matrix inversion only allowed for square matrices
        # so assert the last two dimensions are equal
        type_check.expect(a_type.shape[-1] == a_type.shape[-2])

    @precision._fp16_mixed_precision_helper
    def forward_cpu(self, x):
        self.retain_outputs((0,))
        try:
            invx = utils.force_array(numpy.linalg.inv(x[0]))
        except numpy.linalg.LinAlgError:
            raise ValueError('Input has singular matrices.')
        return invx,

    @precision._fp16_mixed_precision_helper
    def forward_gpu(self, x):
        self.retain_outputs((0,))
        invx, info = _inv_gpu(x[0])
        if chainer.is_debug():
            if cuda.cupy.any(info != 0):
                raise ValueError('Input has singular matrices.')
        return invx,

    def backward(self, x, gy):
        invx, = self.get_retained_outputs()
        # Unpack 1-length tuples
        gy, = gy
        # Gradient is - x^-T (dx) x^-T
        ret = chainer.functions.matmul(-invx, gy, transa=True)
        ret2 = chainer.functions.matmul(ret, invx, transb=True)
        return ret2,


def inv(a):
    """Computes the inverse of square matrix.

        a (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input array to compute the inverse for. Shape of
            the array should be ``(n, n)`` where ``n`` is the dimensionality of
            a square matrix.

    Returns:
        ~chainer.Variable: Matrix inverse of ``a``.
    """
    return Inv().apply((a,))[0]


def batch_inv(a):
    """Computes the inverse of a batch of square matrices.

    Args:
        a (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input array to compute the inverse for. Shape of
            the array should be ``(m, n, n)`` where ``m`` is the number of
            matrices in the batch, and ``n`` is the dimensionality of a square
            matrix.

    Returns:
        ~chainer.Variable: Inverse of every matrix in the batch of matrices.
    """
    return BatchInv().apply((a,))[0]
