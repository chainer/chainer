import numpy

from chainer import cuda
from chainer import function
from chainer.utils import array
from chainer.utils import type_check


class L2Normalization(function.Function):

    """L2 normalization"""

    def __init__(self, eps=1e-5):
        self.eps = eps

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 2,
        )

    def forward_cpu(self, inputs):
        x = array.as_mat(inputs[0])
        norm = numpy.linalg.norm(x, axis=1) + self.eps
        return x / norm[:, numpy.newaxis],

    def forward_gpu(self, inputs):
        x = array.as_mat(inputs[0])
        l2norm_kernel = cuda.cupy.ReductionKernel(
            'T x', 'T y', 'x * x', 'a + b', 'y = sqrt(a)', '0', 'l2norm'
        )
        norm = cuda.cupy.broadcast_to(
            l2norm_kernel(x, axis=1).reshape(-1, 1),
            x.shape
        )

        return x / norm,

    def backward(self, inputs, gy):
        x = inputs[0]
        gy = gy[0]

        xp = cuda.get_array_module(x)

        if xp is numpy:
            norm = numpy.sqrt((x * x).sum(axis=1)) + self.eps
            norm = norm[:, numpy.newaxis]

            gx = gy * norm - (x * gy).sum(axis=1)[:, numpy.newaxis] * x / norm
            gx = gx / norm**2
        else:
            l2norm_kernel = cuda.cupy.ReductionKernel(
                'T x', 'T y', 'x * x', 'a + b', 'y = sqrt(a)', '0', 'l2norm'
            )
            norm = cuda.cupy.broadcast_to(
                l2norm_kernel(x, axis=1).reshape(-1, 1),
                x.shape
            )
            x_gy = cuda.cupy.broadcast_to(
                (x*gy).sum(axis=1, keepdims=True),
                x.shape
            )
            gx = cuda.elementwise(
                'T gy, T x, T x_gy, T norm',
                'T gx',
                'gx = (gy * norm - x_gy * x / norm) / (norm * norm)',
                'l2_bwd')(gy, x, x_gy, norm)

        return gx,


def l2_normalization(x, eps=1e-5):
    """L2 norm (a.k.a. Euclidean norm) squared.

    This function implements L2 normalization on a 1D vector. No reduction
    along batch axis is done.  Let :math:`x` be an input vector of dimension
    :math:`(N,D)`, where :math:`N` and :math:`K` denote mini-batchsize and the
    dimension of the input variable. Then, this function computes an output
    vector :math:`y` by following formula:

    .. math::
       y_i = {x_i \\over \\| x_i \\|_2}

    Args:
        x (~chainer.Variable): Two dimensional output variable. The first
            dimension is assumed to be the *minibatch dimension*.
        eps (float): Epsilon value for numerical stability.

    Returns:
        ~chainer.Variable: Two dimensional output variable.

    """
    return L2Normalization(eps)(x)
