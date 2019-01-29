import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
import chainerx


class Sqrt(function_node.FunctionNode):

    @property
    def label(self):
        return 'sqrt'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_chainerx(self, x):
        return chainerx.sqrt(x[0]),

    def forward(self, x):
        self.retain_outputs((0,))
        xp = backend.get_array_module(*x)
        return utils.force_array(xp.sqrt(x[0], dtype=x[0].dtype)),

    def backward(self, indexes, grad_outputs):
        gx = self.get_retained_outputs()[0]
        gy = grad_outputs[0]
        return gy / (gx * 2.0),


class RsqrtGPU(function_node.FunctionNode):

    @property
    def label(self):
        return 'rsqrt'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_gpu(self, inputs):
        self.retain_outputs((0,))
        x, = inputs
        out = cuda.cupyx.rsqrt(x, dtype=x.dtype)
        return utils.force_array(out),

    def backward(self, indexes, grad_outputs):
        y, = self.get_retained_outputs()
        gy, = grad_outputs
        return gy * (y ** 3) * -0.5,


def sqrt(x):
    """Elementwise square root function.

    .. math::
       y_i = \\sqrt x_i.

    If the value of :math:`x_i` is negative, it returns ``Nan`` for :math:`y_i`
    respect to underlying numpy and cupy specification.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Sqrt().apply((x,))[0]


def rsqrt(x):
    """Computes elementwise reciprocal of square root of input :math:`x_i`.

    .. math::
       y_i = {1 \\over \\sqrt x_i}.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :func:`~chainer.functions.sqrt`
    """
    xp = backend.get_array_module(x)
    if xp is numpy or xp is chainerx:
        return 1.0 / sqrt(x)

    # CuPy provides `rsqrt` which is faster than `1.0 / sqrt(x)`.
    return RsqrtGPU().apply((x,))[0]
