import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class Sqrt(function_node.FunctionNode):

    @property
    def label(self):
        return 'sqrt'

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, x):
        self.retain_outputs((0,))
        xp = cuda.get_array_module(*x)
        return utils.force_array(xp.sqrt(x[0], dtype=x[0].dtype)),

    def backward(self, indexes, grad_outputs):
        gx = self.get_retained_outputs()[0]
        gy = grad_outputs[0]
        return gy / (gx * 2.0),


class Rsqrt(function_node.FunctionNode):

    @property
    def label(self):
        return 'rsqrt'

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, inputs):
        self.retain_inputs((0,))
        x, = inputs
        xp = cuda.get_array_module(x)
        dtype = x.dtype
        if xp is numpy:
            out = xp.reciprocal(xp.sqrt(x, dtype=dtype), dtype=dtype)
        else:
            # CuPy provides `rsqrt` which is faster than `1.0 / sqrt(x)`.
            out = cuda.cupyx.rsqrt(x, dtype=dtype)
        return utils.force_array(out),

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        gy, = grad_outputs
        return gy * (x ** -1.5) * -0.5,


def sqrt(x):
    """Elementwise square root function.

    .. math::
       y_i = \\sqrt x_i.

    If the value of :math:`x_i` is negative, it returns ``Nan`` for :math:`y_i`
    respect to underlying numpy and cupy specification.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Sqrt().apply((x,))[0]


def rsqrt(x):
    """Computes elementwise reciprocal of square root of input :math:`x_i`.

    .. math::
       y_i = {1 \\over \\sqrt x_i}.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :func:`~chainer.functions.sqrt`
    """
    return Rsqrt().apply((x,))[0]
