import numpy

from chainer import cuda
from chainer import function
from chainer.functions import sigmoid
from chainer import utils
from chainer.utils import type_check


class SigmoidCrossEntropy(function.Function):

    """Sigmoid activation followed by a sigmoid cross entropy loss."""

    ignore_label = -1

    def check_type_forward(self, in_types):
        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.int32,
            x_type.shape == t_type.shape
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        self.ignore_mask = (t != self.ignore_label)
        # stable computation of the cross entropy.
        loss = -(
            self.ignore_mask *
            (x * (t - (x >= 0)) - xp.log1p(xp.exp(-xp.abs(x)))))

        return utils.force_array(loss.astype(x.dtype)),

    def backward(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        y, = sigmoid.Sigmoid().forward((x,))
        gx = (gloss * self.ignore_mask * (y - t)).astype(y.dtype)
        return gx, None


def sigmoid_cross_entropy(x, t):
    """Computes cross entropy loss for pre-sigmoid activations.

    Args:
        x (Variable): A variable object holding an array matrix whose elements
            indicate the unnormalized log probabilities.
        t (Variable): Variable holding an int32 vector of ground truth labels.
            If ``t[i] == -1``, corresponding ``x[i]`` is ignored.
            Loss is zero if all ground truth labels are ``-1``.

    Returns:
        Variable: A variable object holding an array of the cross entropy.
            The shape is same as ``x``. The value of the ignored instance
            is set to ``0``.

    .. note::

       This function is differentiable only by ``x``.

    """
    return SigmoidCrossEntropy()(x, t)
