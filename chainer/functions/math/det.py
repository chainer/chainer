import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer.functions.math import matmul
from chainer import utils
from chainer.utils import type_check


def _det_gpu(b):
    # We do a batched LU decomposition on the GPU to compute
    # and compute the determinant by multiplying the diagonal.
    # Change the shape of the array to be size=1 minibatch if necessary.
    # Also copy the matrix as the elments will be modified in-place.
    a = matmul._as_batch_mat(b).copy()
    n = a.shape[1]
    n_matrices = len(a)
    # Pivot array
    p = cuda.cupy.zeros((n_matrices, n), dtype='int32')
    # Output array
    # These arrays hold information on the execution success
    # or if the matrix was singular.
    info = cuda.cupy.zeros(n_matrices, dtype=numpy.intp)
    ap = matmul._mat_ptrs(a)
    _, lda = matmul._get_ld(a)
    cuda.cublas.sgetrfBatched(cuda.Device().cublas_handle, n, ap.data.ptr, lda,
                              p.data.ptr, info.data.ptr, n_matrices)
    det = cuda.cupy.prod(a.diagonal(axis1=1, axis2=2), axis=1)
    # The determinant is equal to the product of the diagonal entries
    # of `a` where the sign of `a` is flipped depending on whether
    # the pivot array is equal to its index.
    rng = cuda.cupy.arange(1, n + 1, dtype='int32')
    parity = cuda.cupy.sum(p != rng, axis=1) % 2
    sign = 1. - 2. * parity.astype('float32')
    return det * sign, info


class BatchDet(function_node.FunctionNode):

    @property
    def label(self):
        return 'det'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        a_type, = in_types
        type_check.expect(a_type.dtype.kind == 'f')
        # Only a minibatch of 2D array shapes allowed.
        type_check.expect(a_type.ndim == 3)
        # Matrix inversion only allowed for square matrices
        # so assert the last two dimensions are equal.
        type_check.expect(a_type.shape[-1] == a_type.shape[-2])

    def forward_cpu(self, x):
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        detx = utils.force_array(numpy.linalg.det(x[0]))
        return detx,

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        detx, _ = _det_gpu(x[0])
        return detx,

    def backward(self, indexes, gy):
        x, = self.get_retained_inputs()
        detx, = self.get_retained_outputs()
        gy, = gy
        inv_x = chainer.functions.batch_inv(
            chainer.functions.transpose(x, (0, 2, 1)))
        gy = chainer.functions.broadcast_to(gy[:, None, None], inv_x.shape)
        detx = chainer.functions.broadcast_to(detx[:, None, None], inv_x.shape)
        grad = gy * detx * inv_x
        return grad,


def batch_det(a):
    """Computes the determinant of a batch of square matrices.

    Args:
        a (Variable): Input array to compute the determinant for.
        The first dimension should iterate over each matrix and be
        of the batchsize.

    Returns:
        ~chainer.Variable: vector of determinants for every matrix
        in the batch.

    """
    return BatchDet().apply((a,))[0]


def det(a):
    """Computes the determinant of a single square matrix.

    Args:
        a (Variable): Input array to compute the determinant for.

    Returns:
        ~chainer.Variable: Scalar determinant of the matrix a.

    """
    shape = (1, a.shape[0], a.shape[1])
    batched_a = chainer.functions.reshape(a, shape)
    batched_det = BatchDet().apply((batched_a,))[0]
    return chainer.functions.reshape(batched_det, ())
