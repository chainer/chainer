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
        type_check.expect(in_types.size() == 2)

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

        return utils.force_array(loss, dtype=x.dtype),

    def backward(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        y, = sigmoid.Sigmoid().forward((x,))
        gx = (gloss * self.ignore_mask * (y - t))
        gx = utils.force_array(gx, dtype=y.dtype)
        return gx, None


def sigmoid_cross_entropy(x, t):
    """Computes cross entropy loss for pre-sigmoid activations.

    Args:
        x (Variable): Variable holding a float32 array/scalar whose element(s)
            indicate the unnormalized log probabilities.
        t (Variable): Variable holding an int32 array/scalar of ground truth
            label(s).
            If ``t[i0, i1, ... ] == -1``, corresponding ``x[i0, i1, ... ]`` is
            ignored. In this case, the loss value of the ignored instance is
            set to ``0``.

    Returns:
        Variable: Variable holding a float32 array of the cross entropy.
            The shape is same as that of ``x``.

    .. note::

       This function is differentiable only by ``x``.

    """
    return SigmoidCrossEntropy()(x, t)
