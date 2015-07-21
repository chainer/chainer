import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class CrossCovariance(function.Function):

    """Cross-covariance loss."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn
        self.y_centered = None
        self.z_centered = None
        self.covariance = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        z_type, y_type = in_types

        type_check.expect(
            z_type.dtype == numpy.float32,
            z_type.ndim == 2,
            y_type.dtype == numpy.float32,
            y_type.ndim == 2,

            z_type.shape[0] == y_type.shape[0]
        )

    def forward_cpu(self, inputs):
        y, z = inputs
        y_mean = y.mean(axis=0, keepdims=True)
        z_mean = z.mean(axis=0, keepdims=True)
        self.y_centered = (y - y_mean)[:, :, None]
        self.z_centered = (z - z_mean)[:, None, :]
        self.covariance = (self.y_centered * self.z_centered).mean(axis=0)
        cost = 0.5 * (self.covariance**2).sum(keepdims=True)
        return cost.reshape(()),

    def forward_gpu(self, inputs):
        y, z = inputs

        # Center inputs
        y_mean = cuda.empty((1, y.shape[1]))
        z_mean = cuda.empty((1, z.shape[1]))
        cuda.cumisc.mean(y, axis=0, out=y_mean, keepdims=True)
        cuda.cumisc.mean(z, axis=0, out=z_mean, keepdims=True)
        self.y_centered = cuda.cumisc.subtract(y, y_mean)
        self.z_centered = cuda.cumisc.subtract(z, z_mean)

        # Calculate cross-covariance
        self.covariance = cuda.empty((y.shape[1], z.shape[1]))
        cuda.culinalg.add_dot(self.y_centered, self.z_centered, self.covariance, transa='T',
                              alpha=1./y.shape[0], beta=0.)

        # Calculate cost
        cost = cuda.cumisc.sum(0.5 * self.covariance**2)
        return cost,

    def backward_cpu(self, inputs, grad_outputs):
        y, z = inputs
        N = numpy.asarray(y.shape[0], dtype=numpy.float32)
        gy = self.covariance[None, :, :] * 1/N * self.z_centered
        gy = gy.sum(axis=-1)
        gz = self.covariance[None, :, :] * 1/N * self.y_centered
        gz = gz.sum(axis=-2)
        return gy, gz

    def backward_gpu(self, inputs, grad_outputs):
        y, z = inputs
        N = y.shape[0]
        # TODO: Dirty hack. Sometimes type(y.shape) is list which causes an error in scikit-cuda because it checks if
        #   c_gpu.shape == (m, n) which is a tuple.
        y.shape = tuple(y.shape)
        z.shape = tuple(z.shape)
        gy = cuda.empty(y.shape)
        gz = cuda.empty(z.shape)
        cuda.culinalg.add_dot(self.z_centered, self.covariance, gy, transb='T', alpha=1./N, beta=0.)
        cuda.culinalg.add_dot(self.y_centered, self.covariance,gz, alpha=1./N, beta=0.)
        gy.shape = list(gy.shape)
        gz.shape = list(gz.shape)
        return gy, gz


def cross_covariance(y, z, use_cudnn=True):
    """Computes the sum-squared cross-covariance penalty between ``y`` and ``z``

    Args:
        y (Variable): Variable holding a matrix with rows as trails
        z (Variable): Variable holding a matrix with rows as trails

    Returns:
        Variable: A variable holding a scalar of the cross covariance loss.

    .. note::

       This cost can be used to disentangle variables.
       See http://arxiv.org/abs/1412.6583v3 for details.

    """
    return CrossCovariance(use_cudnn)(y, z)
