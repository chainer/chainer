import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer import utils
from chainer.utils import type_check


class BatchLogdet(function_node.FunctionNode):

    @property
    def label(self):
        return 'logdet'

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
        sign, logdetx = utils.force_array(numpy.linalg.slogdet(x[0]))
        if (sign <= 0).all():
            raise ValueError('Determinant is not positive')
        return logdetx,

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        sign, logdetx = utils.force_array(cuda.cupy.linalg.slogdet(x[0]))
        if (sign <= 0).all():
            raise ValueError('Determinant is not positive')
        return logdetx,

    def backward(self, indexes, gy):
        x, = self.get_retained_inputs()
        logdetx, = self.get_retained_outputs()
        gy, = gy
        inv_x = chainer.functions.batch_inv(
            chainer.functions.transpose(x, (0, 2, 1)))
        gy = chainer.functions.broadcast_to(gy[:, None, None], inv_x.shape)
        grad = gy * inv_x
        return grad,


def batch_logdet(a):
    """Computes the logarithm of determinant of a batch of square matrices.

    Args:
        a (Variable): Input array to compute the logarithm of determinant for.
        The first dimension should iterate over each matrix and be
        of the batchsize.

    Returns:
        ~chainer.Variable: vector of logarithm of determinants for every matrix
        in the batch.

    """
    return BatchLogdet().apply((a,))[0]


def logdet(a):
    """Computes the logarithm of determinant of a single square matrix.

    Args:
        a (Variable): Input array to compute the logarithm of determinant for.

    Returns:
        ~chainer.Variable: Scalar logarithm of determinant of the matrix a.

    """
    shape = (1, a.shape[0], a.shape[1])
    batched_a = chainer.functions.reshape(a, shape)
    batched_logdet = BatchLogdet().apply((batched_a,))[0]
    return chainer.functions.reshape(batched_logdet, ())
