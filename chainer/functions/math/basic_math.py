import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer.functions.math import matmul as _matmul
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
    elif isinstance(value, (numpy.ndarray, cuda.ndarray)):
        return 'constant array'
    else:
        raise ValueError(
            'Value must be a scalar, `numpy.ndarray`, `cupy.ndarray` '
            'or a `Variable`.\nActual: {}'.format(type(value)))


def _check_constant_type(value):
    if numpy.isscalar(value):
        return
    elif isinstance(value, (numpy.ndarray, cuda.ndarray)):
        return
    else:
        raise ValueError(
            'Value must be a scalar, `numpy.ndarray`, `cupy.ndarray` '
            'or a `Variable`.\nActual: {}'.format(type(value)))


def _preprocess_const(x, value):
    xp = cuda.get_array_module(x)
    if not numpy.isscalar(value) and cuda.get_array_module(value) != xp:
        # TODO(unno): We can transfer arrays automatically
        raise TypeError('Cannot mix cupy.ndarray and numpy.ndarray')

    b = xp.broadcast(x, value)
    if b.shape != x.shape:
        raise ValueError('Failed to broadcast arrays')
    return utils.force_type(x.dtype, value)


class Neg(function_node.FunctionNode):

    @property
    def label(self):
        return '__neg__'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

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

    @property
    def label(self):
        return '|_|'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs((0,))
        return utils.force_array(abs(x[0])),

    def backward(self, indexes, grad_outputs):
        x = self.get_retained_inputs()[0]
        return AbsoluteGrad(x.data).apply(grad_outputs)


class AbsoluteGrad(function_node.FunctionNode):

    def __init__(self, x):
        super(AbsoluteGrad, self).__init__()
        self.x = x

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
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

    @property
    def label(self):
        return '_ + _'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, x):
        y = utils.force_array(x[0] + x[1])
        return y,

    def backward(self, indexes, gy):
        return gy[0], gy[0]


class AddConstant(function_node.FunctionNode):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ + %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        value = _preprocess_const(x[0], self.value)
        return utils.force_array(x[0] + value),

    def backward(self, indexes, gy):
        return gy[0],


def add(self, rhs):  # lhs + rhs
    """Element-wise addition.

    Returns:
        ~chainer.Variable: Output variable.
    """
    if isinstance(rhs, variable.Variable):
        return Add().apply((self, rhs))[0]
    _check_constant_type(rhs)
    return AddConstant(rhs).apply((self,))[0]


class Sub(function_node.FunctionNode):

    @property
    def label(self):
        return '_ - _'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, x):
        return utils.force_array(x[0] - x[1]),

    def backward(self, indexes, gy):
        return gy[0], -gy[0]


def sub(self, rhs):  # lhs - rhs
    """Element-wise subtraction.

    Returns:
        ~chainer.Variable: Output variable.
    """

    if isinstance(rhs, variable.Variable):
        return Sub().apply((self, rhs))[0]
    _check_constant_type(rhs)
    return AddConstant(-rhs).apply((self,))[0]


class SubFromConstant(function_node.FunctionNode):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '%s - _' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        self.retain_inputs(())
        value = _preprocess_const(x[0], self.value)
        return utils.force_array(value - x[0]),

    def backward(self, indexes, gy):
        return -gy[0],


def rsub(self, rhs):  # rhs - lhs
    """Element-wise subtraction.

    Returns:
        ~chainer.Variable: Output variable.
    """
    if isinstance(rhs, variable.Variable):
        return Sub().apply((rhs, self))[0]
    _check_constant_type(rhs)
    return SubFromConstant(rhs).apply((self,))[0]


class Mul(function_node.FunctionNode):

    @property
    def label(self):
        return '_ * _'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, x):
        self.retain_inputs((0, 1))
        return utils.force_array(x[0] * x[1]),

    def backward(self, indexes, gy):
        xs = self.get_retained_inputs()
        return tuple(gy[0] * xs[1 - i] for i in indexes)


class MulConstant(function_node.FunctionNode):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ * %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        value = _preprocess_const(x[0], self.value)
        return utils.force_array(value * x[0]),

    def backward(self, indexes, gy):
        return self.value * gy[0],


def mul(self, rhs):  # lhs * rhs
    """Element-wise multiplication.

    Returns:
        ~chainer.Variable: Output variable.
    """

    if isinstance(rhs, variable.Variable):
        return Mul().apply((self, rhs))[0]
    _check_constant_type(rhs)
    return MulConstant(rhs).apply((self,))[0]


class Div(function_node.FunctionNode):

    @property
    def label(self):
        return '_ / _'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, x):
        self.retain_inputs((0, 1))
        return utils.force_array(x[0] / x[1]),

    def backward(self, indexes, grad_outputs):
        x = self.get_retained_inputs()
        return DivGrad().apply((x[0], x[1], grad_outputs[0]))


class DivGrad(function_node.FunctionNode):

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        x0, x1, gy = inputs
        gx0 = utils.force_array(gy / x1)
        return gx0, utils.force_array(-gx0 * x0 / x1)

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        x0, x1, gy = inputs
        return cuda.elementwise(
            'T x0, T x1, T gy',
            'T gx0, T gx1',
            '''
               gx0 = gy / x1;
               gx1 = -gx0 * x0 / x1;
            ''', 'div_bwd')(x0, x1, gy)

    def backward(self, indexes, grad_outputs):
        x0, x1, gy = self.get_retained_inputs()
        ret = []
        x1_square = x1 * x1
        if 0 in indexes:
            ggx0 = - grad_outputs[1] * gy / x1_square
            ret.append(ggx0)
        if 1 in indexes:
            ggx1 = \
                - grad_outputs[0] * gy / x1_square + \
                grad_outputs[1] * 2 * gy * x0 / (x1_square * x1)
            ret.append(ggx1)
        if 2 in indexes:
            ggy = grad_outputs[0] / x1 - grad_outputs[1] * x0 / x1_square
            ret.append(ggy)
        return ret


def div(self, rhs):  # lhs / rhs
    """Element-wise division

    Returns:
        ~chainer.Variable: Output variable.
    """

    if isinstance(rhs, variable.Variable):
        return Div().apply((self, rhs))[0]
    _check_constant_type(rhs)
    return MulConstant(1. / rhs).apply((self,))[0]


class DivFromConstant(function_node.FunctionNode):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '%s / _' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
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
        gx = cuda.elementwise('T x, T gy, T value', 'T gx',
                              'gx = -value * gy / (x * x)',
                              'div_from_const_bwd')(x, gy, value)
        return gx,

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

    if isinstance(rhs, variable.Variable):
        return Div().apply((rhs, self))[0]
    _check_constant_type(rhs)
    return DivFromConstant(rhs).apply((self,))[0]


class PowVarVar(function_node.FunctionNode):

    @property
    def label(self):
        return '_ ** _'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, x):
        self.retain_inputs((0, 1))
        self.y = x[0] ** x[1]
        return utils.force_array(self.y),

    def backward(self, indexes, gy):
        inputs = self.get_retained_inputs()
        return PowVarVarGrad(self.y).apply((inputs[0], inputs[1], gy[0]))


class PowVarVarGrad(function_node.FunctionNode):

    def __init__(self, y):
        self.y = y

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape,
            in_types[0].dtype == in_types[2].dtype,
            in_types[0].shape == in_types[2].shape,
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1, 2))
        x0, x1, gy = inputs

        one = x1.dtype.type(1)
        gx0 = utils.force_array(x1 * (x0 ** (x1 - one)) * gy)
        gx1 = utils.force_array(numpy.log(x0) * self.y * gy)
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
            gx0 = (ggx0 * x1 * (x1 - 1) * pow_x0_x1_2 +
                   ggx1 * pow_x0_x1_1 * (log_x0 * x1 + 1)) * gy
            ret.append(gx0)
        if 1 in indexes:
            gx1 = (ggx0 * pow_x0_x1_1 * (log_x0 * x1 + 1) +
                   ggx1 * log_x0 * log_x0 * pow_x0_x1) * gy
            ret.append(gx1)
        if 2 in indexes:
            ggy = ggx0 * x1 * pow_x0_x1_1 + ggx1 * log_x0 * pow_x0_x1
            ret.append(ggy)
        return ret


class PowVarConst(function_node.FunctionNode):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ ** %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        self.retain_inputs((0,))
        y = x[0] ** _preprocess_const(x[0], self.value)
        return utils.force_array(y, x[0].dtype),

    def backward(self, indexes, gy):
        inputs = self.get_retained_inputs()
        return PowVarConstGrad(self.value).apply((inputs[0], gy[0]))


class PowVarConstGrad(function_node.FunctionNode):

    def __init__(self, value):
        self.value = value
        self.val = self.val_1 = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
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

    if isinstance(rhs, variable.Variable):
        return PowVarVar().apply((self, rhs))[0]
    _check_constant_type(rhs)
    return PowVarConst(rhs).apply((self,))[0]


class PowConstVar(function_node.FunctionNode):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '%s ** _' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
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

    def __init__(self, value):
        self.value = value

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        y, gy = inputs

        self.value = _preprocess_const(y, self.value)
        gx = utils.force_array(
            numpy.log(self.value, dtype=y.dtype) * y * gy)
        return gx,

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        y, gy = inputs

        self.value = _preprocess_const(y, self.value)
        gx = cuda.elementwise(
            'T y, T gy, T value', 'T gx',
            'gx = log(value) * y * gy',
            'pow_const_var_bwd')(y, gy, self.value)
        return gx,

    def backward(self, indexes, ggx):
        y, gy = self.get_retained_inputs()

        xp = cuda.get_array_module(y)
        gygy = xp.log(self.value) * ggx[0]

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

    if isinstance(rhs, variable.Variable):
        return PowVarVar().apply((rhs, self))[0]
    _check_constant_type(rhs)
    return PowConstVar(rhs).apply((self,))[0]


class MatMulVarVar(_matmul.MatMul):

    @property
    def label(self):
        return '_ @ _'


class MatMulVarConst(function_node.FunctionNode):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ @ %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        a_type = in_types[0]
        b_type = self.value

        type_check.expect(
            a_type.dtype.kind == 'f',
            b_type.dtype.kind == 'f',
            a_type.ndim >= 1,
            a_type.ndim == b_type.ndim,
        )

        ndim = type_check.eval(a_type.ndim)
        if ndim == 1:
            type_check.expect(a_type.shape == b_type.shape)
        else:
            a_idx = _matmul._get_check_index(False, False,
                                             row_idx=-2, col_idx=-1)
            b_idx = _matmul._get_check_index(False, True,
                                             row_idx=-2, col_idx=-1)
            type_check.expect(
                a_type.shape[:-2] == b_type.shape[:-2],
                a_type.shape[a_idx] == b_type.shape[b_idx],
            )

    def forward(self, x):
        self.retain_inputs((0,))
        return utils.force_array(_matmul._matmul(x[0], self.value)),

    def backward(self, indexes, gy):
        x = self.get_retained_inputs()
        if gy[0].ndim == 0:
            gx0 = chainer.functions.broadcast_to(
                gy[0], self.value.shape) * self.value
        else:
            gx0 = chainer.functions.reshape(
                chainer.functions.matmul(gy[0], self.value, False, True),
                x[0].shape)
        return gx0,


class MatMulConstVar(function_node.FunctionNode):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '%s @ _' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        a_type = self.value
        b_type = in_types[0]

        type_check.expect(
            a_type.dtype.kind == 'f',
            b_type.dtype.kind == 'f',
            a_type.ndim >= 1,
            a_type.ndim == b_type.ndim,
        )

        ndim = type_check.eval(a_type.ndim)
        if ndim == 1:
            type_check.expect(a_type.shape == b_type.shape)
        else:
            a_idx = _matmul._get_check_index(False, False,
                                             row_idx=-2, col_idx=-1)
            b_idx = _matmul._get_check_index(False, True,
                                             row_idx=-2, col_idx=-1)
            type_check.expect(
                a_type.shape[:-2] == b_type.shape[:-2],
                a_type.shape[a_idx] == b_type.shape[b_idx],
            )

    def forward(self, x):
        self.retain_inputs((0,))
        return utils.force_array(_matmul._matmul(self.value, x[0])),

    def backward(self, x, gy):
        x = self.get_retained_inputs()
        if gy[0].ndim == 0:
            gx1 = chainer.functions.broadcast_to(
                gy[0], self.value.shape) * self.value
        else:
            gx1 = chainer.functions.reshape(
                chainer.functions.matmul(self.value, gy[0], True, False),
                x[0].shape)
        return gx1,


def matmul(self, rhs):  # lhs @ rhs
    """Matrix multiplication.

    Returns:
        ~chainer.Variable: Output variable.
    """

    if isinstance(rhs, variable.Variable):
        return MatMulVarVar().apply((self, rhs))[0]
    _check_constant_type(rhs)
    return MatMulVarConst(rhs).apply((self,))[0]


def rmatmul(self, rhs):  # rhs @ lhs
    """Matrix multiplication.

    Returns:
        ~chainer.Variable: Output variable.
    """

    if isinstance(rhs, variable.Variable):
        return MatMulVarVar().apply((rhs, self))[0]
    _check_constant_type(rhs)
    return MatMulConstVar(rhs).apply((self,))[0]


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
    variable.Variable.__pow__ = pow
    variable.Variable.__rpow__ = rpow
    variable.Variable.__matmul__ = matmul
    variable.Variable.__rmatmul__ = rmatmul
