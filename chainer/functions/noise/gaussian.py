import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import argument
from chainer.utils import type_check


class Gaussian(function_node.FunctionNode):

    """Gaussian sampling function.

    .. note::

        In forward calculation, this function takes a mean and the logarithm of
        a variance as inputs, and draws a sample from a Gaussian distribution
        accordingly.

    """

    def __init__(self, eps=None):
        # When ``eps`` is set to None, per-instance noise that is generated
        # once during its first forward pass and then reused in subsequent
        # calls.
        self.eps = eps

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('mean', 'ln_var'))

        m_type, v_type = in_types
        type_check.expect(
            m_type.dtype.kind == 'f',
            m_type.dtype == v_type.dtype,
            m_type.shape == v_type.shape,
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((1,))

        mean, ln_var = inputs
        if self.eps is None:
            self.eps = (
                numpy.random.standard_normal(ln_var.shape)
                .astype(mean.dtype, copy=False)
            )

        self.noise = numpy.exp(ln_var * mean.dtype.type(0.5)) * self.eps
        return utils.force_array(mean + self.noise),

    def forward_gpu(self, inputs):
        self.retain_inputs((1,))

        mean, ln_var = inputs
        if self.eps is None:
            if mean.dtype != numpy.float16:
                self.eps = cuda.cupy.random.standard_normal(
                    ln_var.shape, dtype=mean.dtype)
            else:
                # Draw samples in FP32 then cast them to FP16 because
                # cupy.random does not support FP16 currently.
                self.eps = cuda.cupy.random.standard_normal(
                    ln_var.shape, dtype=numpy.float32).astype(numpy.float16)

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


def gaussian(mean, ln_var, **kwargs):
    """gaussian(mean, ln_var, *, eps=None, return_eps=False)

    Gaussian sampling function.

    This function takes a mean :math:`\\mu` and the logarithm of a variance
    :math:`\\log(\\sigma^2)` as inputs and outputs a sample drawn from a
    Gaussian distribution :math:`N(\\mu, \\sigma)`.

    The inputs must have the same shape.

    Args:
        mean (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable representing the mean :math:`\\mu`.
        ln_var (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable representing the logarithm of a variance
            :math:`\\log(\\sigma^2)`.
        eps (:ref:`ndarray` or None):
            The eps value to be used.
            You do not have to specify this value, unless you need to make
            results deterministic.
            If ``eps`` is not specified or set to ``None``, an eps value will
            be generated randomly.
            The shape and dtype must be the same as ``ln_var`` and should be
            on the same device.
        return_eps (bool):
            If ``True``, the eps value used in this function is returned
            together with the output variable.
            The returned eps can later be reused by passing it to the ``eps``
            argument.

    Returns:
        ~chainer.Variable or tuple:
            When ``return_eps`` is ``False`` (default), returns the output
            variable with the shape of ``mean`` and/or ``ln_var``.
            When ``True``, returns the tuple of the output variable and eps
            (:ref:`ndarray`).
            The eps will be on the same device as the input (``ln_var``).

    """
    eps = None
    return_eps = False
    if kwargs:
        eps, return_eps = argument.parse_kwargs(
            kwargs, ('eps', eps), ('return_eps', return_eps))

    func = Gaussian(eps)
    out = func.apply((mean, ln_var))[0]
    if return_eps:
        return out, func.eps
    return out
