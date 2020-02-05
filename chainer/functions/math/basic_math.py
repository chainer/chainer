import math

import numpy
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import function_node
import chainer.functions
from chainer.functions.math import floor as _floor
from chainer import utils
from chainer.utils import type_check
from chainer import variable


def _convert_value_to_string(value):
    if isinstance(value, variable.Variable):
        value = value.data

    if numpy.isscalar(value):
        if value < 0:
            return '({})'.format(value)
        else:
            return str(value)

    array_types = chainer.get_array_types()
    if isinstance(value, array_types):
        return 'constant array'
    else:
        raise ValueError(
            'Value must be a Variable, scalar, {} or {}. Actual: {}'.format(
                ', '.join([str(at) for at in array_types[:-1]]),
                array_types[-1], type(value)))


def _preprocess_const(x, value):
    return x.dtype.type(value)


def _chainerx_preprocess_const(x, value, label):
    # Allow mixing of numpy/cupy array and chainerx array as long as
    # conversion without copy is possible.
    if isinstance(value, (numpy.ndarray, cuda.ndarray)):
        # TODO(niboshi): force zero-copy
        return backend.to_chx(value)

    if isinstance(value, (six.integer_types, float)):
        return value
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, variable.Variable):
        value = variable.as_array(value)
    utils._check_arrays_forward_compatible((x, value), label)
    return value


def _preprocess_rhs(x, value):
    if isinstance(value, chainer.Variable):
        return value

    if not (numpy.isscalar(value)
            or isinstance(value, chainer.get_array_types())):
        raise TypeError(
            'Value must be a scalar, `numpy.ndarray`, `cupy.ndarray` '
            'or a `Variable`.\nActual: {}'.format(type(value)))

    return value.astype(x.dtype, copy=False)


class Neg(function_node.FunctionNode):

    is_elementwise = True

    @property
    def label(self):
        return '__neg__'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))

    def forward_chainerx(self, x):
        return -x[0],

    def forward(self, x):
        self.retain_inputs(())
        return utils.force_array(-x[0]),

    def backward(self, indexes, gy):
        return -gy[0],


def neg(self):  # -x
    """Element-wise negation.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Neg().apply((self,))[0]


class Absolute(function_node.FunctionNode):

    is_elementwise = True

    @property
    def label(self):
        return '|_|'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs((0,))
        return utils.force_array(abs(x[0])),

    def backward(self, indexes, grad_outputs):
        x = self.get_retained_inputs()[0]
        return AbsoluteGrad(x.data).apply(grad_outputs)


class AbsoluteGrad(function_node.FunctionNode):

    is_elementwise = True

    def __init__(self, x):
        super(AbsoluteGrad, self).__init__()
        self.x = x

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('gy',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, inputs):
        return utils.force_array(numpy.sign(self.x) * inputs[0]),

    def forward_gpu(self, inputs):
        gx0 = cuda.elementwise(
            'T x0, T gy', 'T gx0',
            'gx0 = ((x0 > 0) - (x0 < 0)) * gy',
            'abs_bwd')(self.x, inputs[0])
        return gx0,

    def backward(self, indexes, grad_outputs):
        return AbsoluteGrad(self.x).apply(grad_outputs)


def absolute(self):
    """Element-wise absolute.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Absolute().apply((self,))[0]


class Add(function_node.FunctionNode):

    is_elementwise = True

    @property
    def label(self):
        return '_ + _'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('lhs', 'rhs'))
        type_check.expect(
            in_types[0].dtype == in_types[1].dtype,
        )
        type_check.expect_broadcast_shapes(
            in_types[0].shape, in_types[1].shape)

    def forward_chainerx(self, x):
        return x[0] + x[1],

    def forward(self, x):
        # may broadcast
        y = utils.force_array(x[0] + x[1])
        return y,

    def backward(self, indexes, gy):
        return tuple(chainer.functions.sum_to(gy[0], self.inputs[i].shape)
                     for i in indexes)


class AddConstant(function_node.FunctionNode):

    is_elementwise = True

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ + %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types.size() == 1)

    def forward_chainerx(self, x):
        value = _chainerx_preprocess_const(x[0], self.value, 'add')
        return x[0] + value,

    def forward(self, x):
        value = _preprocess_const(x[0], self.value)
        return utils.force_array(x[0] + value),

    def backward(self, indexes, gy):
        x_node, = self.inputs
        return gy


class MultiAdd(function_node.FunctionNode):

    is_elementwise = True

    def check_type_forward(self, in_types):
        for i, in_type in enumerate(in_types):
            type_check._argname((in_type,), ('x{}'.format(i),))
            type_check.expect(in_types[0].dtype == in_type.dtype)

    def forward(self, xs):
        self.len = len(xs)
        if len(xs) == 1:
            return xs
        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(xs)
                and all(x.shape == xs[0].shape for x in xs[1:])):
            y = intel64.ideep.multi_add(xs)
        else:
            # The output should be a new array. Add the first 2 arrays
            # and get the result y. Then add the rest arrays to y.
            y = xs[0] + xs[1]
            for x in xs[2:]:
                if x.shape == y.shape:
                    y += x
                else:
                    y = x + y

        return utils.force_array(y),

    def backward(self, indexes, gy):
        return tuple(chainer.functions.sum_to(gy[0], x_node.shape)
                     for x_node in self.inputs)


# TODO(hvy): Implement multi-add with chainerx.ndarrays.
def add(*xs):  # lhs + rhs or add more than 2 variables
    """Element-wise addition.

    Returns:
        ~chainer.Variable: Output variable.
    """
    if len(xs) == 2:
        lhs, rhs = xs
        if numpy.isscalar(rhs):
            return AddConstant(rhs).apply((lhs,))[0]
        rhs = _preprocess_rhs(lhs, rhs)
        return Add().apply((lhs, rhs))[0]
    else:
        return MultiAdd().apply(xs)[0]


class Sub(function_node.FunctionNode):

    is_elementwise = True

    @property
    def label(self):
        return '_ - _'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('lhs', 'rhs'))
        type_check.expect(in_types[0].dtype == in_types[1].dtype)
        type_check.expect_broadcast_shapes(
            in_types[0].shape, in_types[1].shape)

    def forward_chainerx(self, x):
        return x[0] - x[1],

    def forward(self, x):
        # may broadcast
        return utils.force_array(x[0] - x[1]),

    def backward(self, indexes, gy):
        x1, x2 = self.inputs
        g, = gy
        return (
            chainer.functions.sum_to(g, x1.shape) if 0 in indexes else None,
            -chainer.functions.sum_to(g, x2.shape) if 1 in indexes else None,
        )


def sub(self, rhs):  # lhs - rhs
    """Element-wise subtraction.

    Returns:
        ~chainer.Variable: Output variable.
    """
    if numpy.isscalar(rhs):
        return AddConstant(-rhs).apply((self,))[0]
    rhs = _preprocess_rhs(self, rhs)
    return Sub().apply((self, rhs))[0]


class SubFromConstant(function_node.FunctionNode):

    is_elementwise = True

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '%s - _' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))

    def forward(self, x):
        value = _preprocess_const(x[0], self.value)
        return utils.force_array(value - x[0]),

    def backward(self, indexes, gy):
        g, = gy
        return -g,


def rsub(self, rhs):  # rhs - lhs
    """Element-wise subtraction.

    Returns:
        ~chainer.Variable: Output variable.
    """
    if numpy.isscalar(rhs):
        return SubFromConstant(rhs).apply((self,))[0]
    rhs = _preprocess_rhs(self, rhs)
    return Sub().apply((rhs, self))[0]


class Mul(function_node.FunctionNode):

    is_elementwise = True

    @property
    def label(self):
        return '_ * _'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('lhs', 'rhs'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
        )
        type_check.expect_broadcast_shapes(
            in_types[0].shape, in_types[1].shape)

    def forward_chainerx(self, x):
        return x[0] * x[1],

    def forward(self, x):
        self.retain_inputs((0, 1))
        # may broadcast
        return utils.force_array(x[0] * x[1]),

    def backward(self, indexes, gy):
        xs = self.get_retained_inputs()
        return tuple(
            chainer.functions.sum_to(gy[0] * xs[1 - i], xs[i].shape)
            for i in indexes
        )


class MulConstant(function_node.FunctionNode):

    is_elementwise = True

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ * %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))

    def forward_chainerx(self, x):
        value = _chainerx_preprocess_const(x[0], self.value, 'mul')
        return x[0] * value,

    def forward(self, x):
        value = _preprocess_const(x[0], self.value)
        return utils.force_array(value * x[0]),

    def backward(self, indexes, gy):
        g, = gy
        return self.value * g,


def mul(self, rhs):  # lhs * rhs
    """Element-wise multiplication.

    Returns:
        ~chainer.Variable: Output variable.
    """
    if numpy.isscalar(rhs):
        return MulConstant(rhs).apply((self,))[0]
    rhs = _preprocess_rhs(self, rhs)
    return Mul().apply((self, rhs))[0]


class Div(function_node.FunctionNode):

    is_elementwise = True

    @property
    def label(self):
        return '_ / _'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('lhs', 'rhs'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
        )
        type_check.expect_broadcast_shapes(
            in_types[0].shape, in_types[1].shape)

    def forward_chainerx(self, x):
        return x[0] / x[1],

    def forward(self, x):
        self.retain_inputs((0, 1))
        # may broadcast
        return utils.force_array(x[0] / x[1]),

    def backward(self, indexes, grad_outputs):
        x0, x1 = self.get_retained_inputs()
        is_grad_elementwise = x0.shape == x1.shape
        divgrad = DivGrad(is_grad_elementwise)
        return divgrad.apply((x0, x1, grad_outputs[0]))


class DivGrad(function_node.FunctionNode):

    def __init__(self, is_elementwise):
        self.is_elementwise = is_elementwise

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        x0, x1, gy = inputs
        gx0 = utils.force_array(gy / x1)
        gx1 = utils.force_array(-gx0 * x0 / x1)
        return utils.sum_to(gx0, x0.shape), utils.sum_to(gx1, x1.shape)

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        x0, x1, gy = inputs
        gx0, gx1 = cuda.elementwise(
            'T x0, T x1, T gy',
            'T gx0, T gx1',
            '''
               gx0 = gy / x1;
               gx1 = -gx0 * x0 / x1;
            ''', 'div_bwd')(x0, x1, gy)
        return utils.sum_to(gx0, x0.shape), utils.sum_to(gx1, x1.shape)

    def backward(self, indexes, grad_outputs):
        x0, x1, gy = self.get_retained_inputs()
        ggx0, ggx1 = grad_outputs

        ret = []
        x1_square = x1 * x1

        if 0 in indexes:
            if ggx1 is None:
                ret.append(None)
            else:
                gx0 = -ggx1 * gy / x1_square
                ret.append(chainer.functions.sum_to(gx0, x0.shape))

        if 1 in indexes:
            gx1 = None if ggx0 is None else -ggx0 * gy / x1_square
            gx1_1 = (None if ggx1 is None else
                     ggx1 * 2 * gy * x0 / (x1_square * x1))
            if gx1 is None:
                gx1 = gx1_1
            elif gx1_1 is not None:
                gx1 += gx1_1
            ret.append(None if gx1 is None else
                       chainer.functions.sum_to(gx1, x1.shape))

        if 2 in indexes:
            ggy = None if ggx0 is None else ggx0 / x1
            ggy_1 = None if ggx1 is None else ggx1 * x0 / x1_square
            if ggy is None:
                ggy = -ggy_1
            elif ggy_1 is not None:
                ggy -= ggy_1
            ret.append(ggy)

        return ret


def div(self, rhs):  # lhs / rhs
    """Element-wise division

    Returns:
        ~chainer.Variable: Output variable.
    """
    if numpy.isscalar(rhs):
        return MulConstant(1. / rhs).apply((self,))[0]
    rhs = _preprocess_rhs(self, rhs)
    return Div().apply((self, rhs))[0]


# TODO(sonots): Support chainerx
class DivFromConstant(function_node.FunctionNode):

    is_elementwise = True

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '%s / _' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs((0,))
        value = _preprocess_const(x[0], self.value)
        return utils.force_array(value / x[0]),

    def backward(self, indexes, grad_outputs):
        x = self.get_retained_inputs()
        return DivFromConstantGrad(self.value).apply((x[0], grad_outputs[0]))


class DivFromConstantGrad(function_node.FunctionNode):

    def __init__(self, value):
        super(DivFromConstantGrad, self).__init__()
        self.value = value

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        value = _preprocess_const(x, self.value)
        return utils.force_array(-value * gy / (x ** 2)),

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        # TODO(beam2d): Make it not use the input
        value = _preprocess_const(x, self.value)
        return cuda.elementwise('T x, T gy, T value', 'T gx',
                                'gx = -value * gy / (x * x)',
                                'div_from_const_bwd')(x, gy, value),

    def backward(self, indexes, grad_outputs):
        x, gy = self.get_retained_inputs()
        value = _preprocess_const(x.data, self.value)
        ret = []
        if 0 in indexes:
            ret.append(grad_outputs[0] * 2 * value * gy / (x ** 3))
        if 1 in indexes:
            ret.append(grad_outputs[0] * -value / (x ** 2))
        return ret


def rdiv(self, rhs):  # rhs / lhs
    """Element-wise division.

    Returns:
        ~chainer.Variable: Output variable.
    """
    if numpy.isscalar(rhs):
        return DivFromConstant(rhs).apply((self,))[0]
    rhs = _preprocess_rhs(self, rhs)
    return Div().apply((rhs, self))[0]


def floordiv(self, rhs):  # lhs // rhs
    """Element-wise floor division.

    Returns:
        ~chainer.Variable: Output variable.
    """

    return _floor.floor(div(self, rhs))


def rfloordiv(self, rhs):  # rhs // lhs
    """Element-wise floor division.

    Returns:
        ~chainer.Variable: Output variable.
    """

    return _floor.floor(rdiv(self, rhs))


class PowVarVar(function_node.FunctionNode):

    is_elementwise = True

    @property
    def label(self):
        return '_ ** _'

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('lhs', 'rhs'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
        )
        type_check.expect_broadcast_shapes(
            in_types[0].shape, in_types[1].shape)

    def forward(self, x):
        self.retain_inputs((0, 1))
        # may broadcast
        self.y = x[0] ** x[1]
        return utils.force_array(self.y),

    def backward(self, indexes, gy):
        x0, x1 = self.get_retained_inputs()
        is_grad_elementwise = x0.shape == x1.shape
        return PowVarVarGrad(
            is_grad_elementwise, self.y).apply((x0, x1, gy[0]))


class PowVarVarGrad(function_node.FunctionNode):

    def __init__(self, is_elementwise, y):
        self.is_elementwise = is_elementwise
        self.y = y

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('lhs', 'rhs', 'gy'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].dtype == in_types[2].dtype,
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        x0, x1, gy = inputs

        one = x1.dtype.type(1)
        gx0 = utils.sum_to(
            utils.force_array(x1 * (x0 ** (x1 - one)) * gy), x0.shape)
        gx1 = utils.sum_to(
            utils.force_array(numpy.log(x0) * self.y * gy), x1.shape)
        return gx0, gx1

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        x0, x1, gy = inputs

        gx0, gx1 = cuda.elementwise(
            'T x0, T x1, T gy, T y', 'T gx0, T gx1',
            '''
            gx0 = x1 * pow(x0, x1 - 1) * gy;
            gx1 = log(x0) * y * gy;
            ''', 'pow_var_var_bwd')(x0, x1, gy, self.y)

        gx0 = utils.sum_to(gx0, x0.shape)
        gx1 = utils.sum_to(gx1, x1.shape)

        return gx0, gx1

    def backward(self, indexes, ggx):
        x0, x1, gy = self.get_retained_inputs()
        ggx0, ggx1 = ggx

        log_x0 = chainer.functions.log(x0)
        pow_x0_x1 = x0 ** x1
        pow_x0_x1_1 = x0 ** (x1 - 1)
        pow_x0_x1_2 = x0 ** (x1 - 2)

        ret = []
        if 0 in indexes:
            gx0_0 = (0 if ggx0 is None else
                     ggx0 * x1 * (x1 - 1) * pow_x0_x1_2)
            gx0_1 = (0 if ggx1 is None else
                     ggx1 * pow_x0_x1_1 * (log_x0 * x1 + 1))
            gx0 = (gx0_0 + gx0_1) * gy
            ret.append(chainer.functions.sum_to(gx0, x0.shape))
        if 1 in indexes:
            gx1_0 = (0 if ggx0 is None else
                     ggx0 * pow_x0_x1_1 * (log_x0 * x1 + 1))
            gx1_1 = (0 if ggx1 is None else
                     ggx1 * log_x0 * log_x0 * pow_x0_x1)
            gx1 = (gx1_0 + gx1_1) * gy
            ret.append(chainer.functions.sum_to(gx1, x1.shape))
        if 2 in indexes:
            ggy_0 = 0 if ggx0 is None else ggx0 * x1 * pow_x0_x1_1
            ggy_1 = 0 if ggx1 is None else ggx1 * log_x0 * pow_x0_x1
            ggy = ggy_0 + ggy_1
            ret.append(ggy)
        return ret


class PowVarConst(function_node.FunctionNode):

    is_elementwise = True

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ ** %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs((0,))
        y = x[0] ** _preprocess_const(x[0], self.value)
        return utils.force_array(y, x[0].dtype),

    def backward(self, indexes, gy):
        inputs = self.get_retained_inputs()
        return PowVarConstGrad(self.value).apply((inputs[0], gy[0]))


class PowVarConstGrad(function_node.FunctionNode):

    is_elementwise = True

    def __init__(self, value):
        self.value = value
        self.val = self.val_1 = None

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x', 'gy'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs

        self.val_1 = _preprocess_const(x, self.value - 1)
        gx = utils.force_type(x.dtype, self.value) * (x ** self.val_1) * gy
        gx = utils.force_array(gx)
        return gx,

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs

        self.val = _preprocess_const(x, self.value)
        gx = cuda.elementwise(
            'T x, T gy, T value', 'T gx',
            'gx = value * pow(x, value - 1) * gy',
            'pow_var_const_bwd')(x, gy, self.val)
        return gx,

    def backward(self, indexes, ggx):
        x, gy = self.get_retained_inputs()

        if self.val is None:
            self.val = _preprocess_const(x.data, self.value)
        if self.val_1 is None:
            self.val_1 = _preprocess_const(x.data, self.value - 1)
        val_2 = _preprocess_const(x.data, self.value - 2)

        ret = []
        if 0 in indexes:
            ret.append(ggx[0] * self.val * gy * self.val_1 * x ** val_2)
        if 1 in indexes:
            ret.append(ggx[0] * self.val * x ** self.val_1)
        return ret


def pow(self, rhs):  # lhs ** rhs
    """Element-wise power function.

    Returns:
        ~chainer.Variable: Output variable.
    """

    if numpy.isscalar(rhs):
        return PowVarConst(rhs).apply((self,))[0]
    rhs = _preprocess_rhs(self, rhs)
    return PowVarVar().apply((self, rhs))[0]


class PowConstVar(function_node.FunctionNode):

    is_elementwise = True

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '%s ** _' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_outputs((0,))
        value = _preprocess_const(x[0], self.value)
        y = value ** x[0]
        return utils.force_array(y),

    def backward(self, indexes, gy):
        outputs = self.get_retained_outputs()
        return PowConstVarGrad(self.value).apply((outputs[0], gy[0]))


class PowConstVarGrad(function_node.FunctionNode):

    is_elementwise = True

    def __init__(self, value):
        self.value = value
        self.log_value = math.log(value)

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('y', 'gy'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        y, gy = inputs

        gx = utils.force_array(y.dtype.type(self.log_value) * y * gy)
        return gx,

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        y, gy = inputs

        value = _preprocess_const(y, self.value)
        gx = cuda.elementwise(
            'T y, T gy, T value', 'T gx',
            'gx = log(value) * y * gy',
            'pow_const_var_bwd')(y, gy, value)
        return gx,

    def backward(self, indexes, ggx):
        y, gy = self.get_retained_inputs()

        gygy = y.dtype.type(self.log_value) * ggx[0]

        ret = []
        if 0 in indexes:
            ret.append(gygy * gy)
        if 1 in indexes:
            ret.append(gygy * y)
        return ret


def rpow(self, rhs):  # rhs ** lhs
    """Element-wise power function.

    Returns:
        ~chainer.Variable: Output variable.
    """

    if numpy.isscalar(rhs):
        return PowConstVar(rhs).apply((self,))[0]
    rhs = _preprocess_rhs(self, rhs)
    return PowVarVar().apply((rhs, self))[0]


def matmul(self, rhs):  # lhs @ rhs
    """Matrix multiplication.

    Returns:
        ~chainer.Variable: Output variable.
    """

    rhs = _preprocess_rhs(self, rhs)
    return chainer.functions.matmul(self, rhs)


def rmatmul(self, rhs):  # rhs @ lhs
    """Matrix multiplication.

    Returns:
        ~chainer.Variable: Output variable.
    """

    rhs = _preprocess_rhs(self, rhs)
    return chainer.functions.matmul(rhs, self)


def install_variable_arithmetics():
    variable.Variable.__neg__ = neg
    variable.Variable.__abs__ = absolute
    variable.Variable.__add__ = add
    variable.Variable.__radd__ = add
    variable.Variable.__sub__ = sub
    variable.Variable.__rsub__ = rsub
    variable.Variable.__mul__ = mul
    variable.Variable.__rmul__ = mul
    variable.Variable.__div__ = div
    variable.Variable.__truediv__ = div
    variable.Variable.__rdiv__ = rdiv
    variable.Variable.__rtruediv__ = rdiv
    variable.Variable.__floordiv__ = floordiv
    variable.Variable.__rfloordiv__ = rfloordiv
    variable.Variable.__pow__ = pow
    variable.Variable.__rpow__ = rpow
    variable.Variable.__matmul__ = matmul
    variable.Variable.__rmatmul__ = rmatmul
