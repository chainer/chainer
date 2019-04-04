import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer import utils
from chainer.utils import type_check


class Sin(function_node.FunctionNode):

    @property
    def label(self):
        return 'sin'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs((0,))
        xp = backend.get_array_module(*x)
        return utils.force_array(xp.sin(x[0])),

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        return SinGrad().apply((x, grad_outputs[0]))


class SinGrad(function_node.FunctionNode):

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        gx = utils.force_array(numpy.cos(x))
        gx *= gy
        return gx,

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        gx = cuda.elementwise(
            'T x, T gy', 'T gx', 'gx = cos(x) * gy', 'sin_bwd'
        )(x, gy)
        return gx,

    def backward(self, indexes, grad_outputs):
        x, gy = self.get_retained_inputs()
        ret = []
        if 0 in indexes:
            ret.append(- sin(x) * gy * grad_outputs[0])
        if 1 in indexes:
            ret.append(cos(x) * grad_outputs[0])
        return ret


def sin(x):
    """Elementwise sin function.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Sin().apply((x,))[0]


class Cos(function_node.FunctionNode):

    @property
    def label(self):
        return 'cos'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs((0,))
        xp = backend.get_array_module(*x)
        return utils.force_array(xp.cos(x[0])),

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        return CosGrad().apply((x, grad_outputs[0]))


class CosGrad(function_node.FunctionNode):

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        gx = utils.force_array(numpy.sin(x))
        numpy.negative(gx, out=gx)
        gx *= gy
        return gx,

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        gx = cuda.elementwise(
            'T x, T gy', 'T gx', 'gx = -sin(x) * gy', 'cos_bwd'
        )(x, gy)
        return gx,

    def backward(self, indexes, grad_outputs):
        x, gy = self.get_retained_inputs()
        ret = []
        if 0 in indexes:
            ret.append(- cos(x) * gy * grad_outputs[0])
        if 1 in indexes:
            ret.append(- sin(x) * grad_outputs[0])
        return ret


def cos(x):
    """Elementwise cos function.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Cos().apply((x,))[0]


class Tan(function_node.FunctionNode):

    @property
    def label(self):
        return 'tan'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs((0,))
        xp = backend.get_array_module(*x)
        return utils.force_array(xp.tan(x[0])),

    def backward(self, indexes, grad_outputs):
        x, = self.get_retained_inputs()
        return grad_outputs[0] / chainer.functions.square(cos(x)),


def tan(x):
    """Elementwise tan function.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Tan().apply((x,))[0]


class Arcsin(function_node.FunctionNode):

    @property
    def label(self):
        return 'arcsin'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs((0,))
        xp = backend.get_array_module(*x)
        return utils.force_array(xp.arcsin(x[0])),

    def backward(self, indexes, grad_outputs):
        x = self.get_retained_inputs()
        return ArcsinGrad().apply((x[0], grad_outputs[0]))


class ArcsinGrad(function_node.FunctionNode):

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        gx = utils.force_array(numpy.square(x))
        numpy.negative(gx, out=gx)
        gx += 1
        numpy.sqrt(gx, out=gx)
        numpy.reciprocal(gx, out=gx)
        gx *= gy
        return gx,

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        gx = cuda.elementwise(
            'T x, T gy', 'T gx',
            'gx = rsqrt((T)1.0 - x * x) * gy',
            'arcsin_bwd'
        )(x, gy)
        return gx,

    def backward(self, indexes, grad_outputs):
        x, gy = self.get_retained_inputs()
        ret = []
        if 0 in indexes:
            ret.append(grad_outputs[0] * gy * x / ((1 - x ** 2) ** 1.5))
        if 1 in indexes:
            ret.append(ArcsinGrad().apply((x, grad_outputs[0]))[0])
        return ret


def arcsin(x):
    """Elementwise arcsine function.

    .. math::
       y_i = \\arcsin x_i.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Arcsin().apply((x,))[0]


class Arccos(function_node.FunctionNode):

    @property
    def label(self):
        return 'arccos'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs((0,))
        xp = backend.get_array_module(*x)
        return utils.force_array(xp.arccos(x[0])),

    def backward(self, indexes, grad_outputs):
        x = self.get_retained_inputs()
        return ArccosGrad().apply((x[0], grad_outputs[0]))


class ArccosGrad(function_node.FunctionNode):

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        gx = utils.force_array(numpy.square(x))
        numpy.negative(gx, out=gx)
        gx += 1
        numpy.sqrt(gx, out=gx)
        numpy.reciprocal(gx, out=gx)
        numpy.negative(gx, out=gx)
        gx *= gy
        return gx,

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        gx = cuda.elementwise(
            'T x, T gy', 'T gx',
            'gx = -rsqrt((T)1.0 - x * x) * gy',
            'arccos_bwd'
        )(x, gy)
        return gx,

    def backward(self, indexes, grad_outputs):
        x, gy = self.get_retained_inputs()
        ret = []
        if 0 in indexes:
            ret.append(- grad_outputs[0] * (gy * x) / ((1 - x ** 2) ** 1.5))
        if 1 in indexes:
            ret.append(ArccosGrad().apply((x, grad_outputs[0]))[0])
        return ret


def arccos(x):
    """Elementwise arccosine function.

    .. math::
       y_i = \\arccos x_i.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Arccos().apply((x,))[0]


class Arctan(function_node.FunctionNode):

    @property
    def label(self):
        return 'arctan'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs((0,))
        xp = backend.get_array_module(*x)
        return utils.force_array(xp.arctan(x[0])),

    def backward(self, indexes, grad_outputs):
        x = self.get_retained_inputs()
        return ArctanGrad().apply((x[0], grad_outputs[0]))


class ArctanGrad(function_node.FunctionNode):

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        gx = utils.force_array(numpy.square(x))
        gx += 1
        numpy.reciprocal(gx, out=gx)
        gx *= gy
        return gx,

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        gx = cuda.elementwise(
            'T x, T gy', 'T gx',
            'gx = (T)1.0 / ((T)1.0 + x * x) * gy',
            'arctan_bwd'
        )(x, gy)
        return gx,

    def backward(self, indexes, grad_outputs):
        x, gy = self.get_retained_inputs()
        ret = []
        x_sq = chainer.functions.square(x)
        if 0 in indexes:
            ret.append(
                -2 * gy * x * grad_outputs[0] /
                (chainer.functions.square(x_sq) + 2 * x_sq + 1))
        if 1 in indexes:
            ret.append(grad_outputs[0] / (x_sq + 1))
        return ret


def arctan(x):
    """Elementwise arctangent function.

    .. math::
       y_i = \\arctan x_i.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Arctan().apply((x,))[0]


class Arctan2(function_node.FunctionNode):

    @property
    def label(self):
        return 'arctan2'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x1', 'x2'))
        type_check.expect(in_types[0].dtype.kind == 'f')
        type_check.expect(in_types[1].dtype.kind == 'f')

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        xp = backend.get_array_module(*inputs)
        x1, x2 = inputs
        return utils.force_array(xp.arctan2(x1, x2)),

    def backward(self, indexes, grad_outputs):
        x1, x2 = self.get_retained_inputs()
        return Arctan2Grad().apply((x1, x2, grad_outputs[0]))


class Arctan2Grad(function_node.FunctionNode):

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        x1, x2, gy = inputs
        sqnorm = x1 ** 2 + x2 ** 2
        gx1 = utils.force_array(x2 / sqnorm * gy)
        gx2 = utils.force_array(-x1 / sqnorm * gy)
        return gx1, gx2

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        x1, x2, gy = inputs
        gx1, gx2 = cuda.elementwise(
            'T x1, T x2, T gy',
            'T gx1, T gx2',
            ('T sqnorm = x1 * x1 + x2 * x2;'
             'gx1 = x2 / sqnorm * gy;'
             'gx2 = -x1 / sqnorm * gy;'),
            'arctan2_bwd'
        )(x1, x2, gy)
        return gx1, gx2

    def backward(self, indexes, grad_outputs):
        x1, x2, gy = self.get_retained_inputs()
        ggx1, ggx2 = grad_outputs
        x1_sq = x1 ** 2
        x2_sq = x2 ** 2
        sqnorm = x1_sq + x2_sq

        ret = []
        if 0 in indexes:
            ret.append(
                (- ggx1 * 2 * x1 * x2 + ggx2 * (x1_sq - x2_sq)) * gy /
                sqnorm ** 2)
        if 1 in indexes:
            ret.append(
                (ggx1 * (x1_sq - x2_sq) + ggx2 * (2 * x1 * x2)) * gy /
                sqnorm ** 2)
        if 2 in indexes:
            ret.append((ggx1 * x2 - ggx2 * x1) / sqnorm)
        return ret


def arctan2(x1, x2):
    """Elementwise arctangent function with two arguments.

    Args:
        x1 (:class:`~chainer.Variable` or :ref:`ndarray`):
            Y-coordinates.
        x2 (:class:`~chainer.Variable` or :ref:`ndarray`):
            X-coordinates.

    Returns:
        ~chainer.Variable: Angles in radians, in the range [-pi, pi].
    """
    return Arctan2().apply((x1, x2))[0]
