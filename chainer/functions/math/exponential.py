import math

import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
import chainerx


class Exp(function_node.FunctionNode):

    @property
    def label(self):
        return 'exp'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_chainerx(self, x):
        return chainerx.exp(x[0]),

    def forward_cpu(self, x):
        self.retain_outputs((0,))
        return utils.force_array(numpy.exp(x[0])),

    def forward_gpu(self, x):
        self.retain_outputs((0,))
        return cuda.cupy.exp(x[0]),

    def backward(self, indexes, gy):
        y = self.get_retained_outputs()[0]
        return y * gy[0],


def exp(x):
    """Elementwise exponential function.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Exp().apply((x,))[0]


class Log(function_node.FunctionNode):

    @property
    def label(self):
        return 'log'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_chainerx(self, x):
        return chainerx.log(x[0]),

    def forward_cpu(self, x):
        self.retain_inputs((0,))
        return utils.force_array(numpy.log(x[0])),

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        return cuda.cupy.log(x[0]),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()[0]
        return utils.force_array(gy[0] / x),


def log(x):
    """Elementwise natural logarithm function.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Log().apply((x,))[0]


class Log2(function_node.FunctionNode):

    @property
    def label(self):
        return 'log2'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, inputs):
        self.retain_inputs((0,))
        x = inputs[0]
        xp = backend.get_array_module(x)
        return utils.force_array(xp.log2(x)),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()[0]
        return gy[0] / x * (1 / math.log(2)),


def log2(x):
    """Elementwise logarithm function to the base 2.

    .. math::
       y_i = \\log_2 x_i.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Log2().apply((x,))[0]


class Log10(function_node.FunctionNode):

    @property
    def label(self):
        return 'log10'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, inputs):
        self.retain_inputs((0,))
        x = inputs[0]
        xp = backend.get_array_module(x)
        return utils.force_array(xp.log10(x)),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()[0]
        return gy[0] / x * (1 / math.log(10)),


def log10(x):
    """Elementwise logarithm function to the base 10.

    .. math::
       y_i = \\log_{10} x_i.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Log10().apply((x,))[0]
