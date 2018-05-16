import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Gaussian(function_node.FunctionNode):

    """Gaussian sampling function.

    .. note::

        In forward calculation, this function takes a mean and the logarithm of
        a variance as inputs, and draws a sample from a Gaussian distribution
        accordingly.

    """

    def __init__(self):
        # Per-instance noise that is generated once during its first forward
        # pass and then reused in subsequent calls, unless explicitly reset
        self.eps = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        m_type, v_type = in_types
        type_check.expect(
            m_type.dtype == numpy.float32,
            v_type.dtype == numpy.float32,
            m_type.shape == v_type.shape,
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((1,))

        mean, ln_var = inputs
        if self.eps is None:
            self.eps = (
                numpy.random.standard_normal(ln_var.shape)
                .astype(numpy.float32)
            )

        self.noise = numpy.exp(ln_var * mean.dtype.type(0.5)) * self.eps
        return utils.force_array(mean + self.noise),

    def forward_gpu(self, inputs):
        self.retain_inputs((1,))

        mean, ln_var = inputs
        if self.eps is None:
            self.eps = cuda.cupy.random.standard_normal(
                ln_var.shape, dtype=mean.dtype)

        self.noise = cuda.cupy.empty_like(mean)
        self.noise = cuda.elementwise(
            'T v, T e', 'T noise',
            'noise = exp(v / 2) * e',
            'gaussian_forward'
        )(ln_var, self.eps)
        return mean + self.noise,

    def backward(self, indexes, grad_outputs):
        ln_var, = self.get_retained_inputs()
        gy, = grad_outputs

        ret = []
        if 0 in indexes:
            ret.append(gy)
        if 1 in indexes:
            noise = chainer.functions.exp(ln_var * 0.5) * self.eps
            ret.append(gy * noise * 0.5)
        return ret


def gaussian(mean, ln_var):
    """Gaussian sampling function.

    This function takes a mean :math:`\\mu` and the logarithm of a variance
    :math:`\\log(\\sigma^2)` as inputs and outputs a sample drawn from a
    Gaussian distribution :math:`N(\\mu, \\sigma)`.

    The inputs must have the same shape.

    Args:
        mean (~chainer.Variable):
            Input variable representing the mean :math:`\\mu`.
        ln_var (~chainer.Variable):
            Input variable representing the logarithm of a variance
            :math:`\\log(\\sigma^2)`.

    Returns:
        ~chainer.Variable:
            Output variable with the shape of ``mean`` and/or ``ln_var``.

    """
    return Gaussian().apply((mean, ln_var))[0]
