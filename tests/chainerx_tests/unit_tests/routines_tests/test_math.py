import chainer
import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import op_utils
from chainerx_tests import math_utils


_in_out_dtypes_arithmetic_invalid = [
    (('bool_', 'bool_'), 'bool_'),
    (('bool_', 'int8'), 'int8'),
    (('bool_', 'int16'), 'int16'),
    (('bool_', 'int32'), 'int32'),
    (('bool_', 'int64'), 'int64'),
    (('bool_', 'uint8'), 'uint8'),
    (('bool_', 'float16'), 'float16'),
    (('bool_', 'float32'), 'float32'),
    (('bool_', 'float64'), 'float64'),
    (('int8', 'bool_'), 'int8'),
    (('int16', 'bool_'), 'int16'),
    (('int32', 'bool_'), 'int32'),
    (('int64', 'bool_'), 'int64'),
    (('uint8', 'bool_'), 'uint8'),
    (('float16', 'bool_'), 'float16'),
    (('float32', 'bool_'), 'float32'),
    (('float64', 'bool_'), 'float64'),
]


_in_out_dtypes_arithmetic = [
    dtypes for dtypes in dtype_utils.result_dtypes_two_arrays
    if dtypes not in _in_out_dtypes_arithmetic_invalid
]


_in_out_dtypes_inplace_arithmetic_invalid = [
    ((t1, t2), t3) for (t1, t2), t3 in _in_out_dtypes_arithmetic
    if (numpy.dtype(t1).kind != 'f' and numpy.dtype(t2).kind == 'f')
] + _in_out_dtypes_arithmetic_invalid


_in_out_dtypes_inplace_arithmetic = [
    dtypes for dtypes in dtype_utils.result_dtypes_two_arrays
    if dtypes not in _in_out_dtypes_inplace_arithmetic_invalid
]


_in_out_dtypes_array_int_scalar = [
    # Int scalar.
    (('int8',), int, 'int8'),
    (('int16',), int, 'int16'),
    (('int32',), int, 'int32'),
    (('int64',), int, 'int64'),
    (('uint8',), int, 'uint8'),
    (('float16',), int, 'float16'),
    (('float32',), int, 'float32'),
    (('float64',), int, 'float64'),
    (('int16',), numpy.int16, 'int16'),
    (('uint8',), numpy.int8, 'uint8'),
    (('float64',), numpy.int8, 'float64'),
    (('float16',), numpy.int64, 'float16'),
]


_in_out_dtypes_int_array_float_scalar = [
    # Int arrays and float scalars.
    (('int8',), float, 'float32'),
    (('int16',), float, 'float32'),
    (('int32',), float, 'float32'),
    (('int64',), float, 'float32'),
    (('uint8',), float, 'float32'),
    (('int8',), numpy.float32, 'float32'),
    (('int64',), numpy.float16, 'float32'),
    (('uint8',), numpy.float64, 'float32'),
]


_in_out_dtypes_float_array_float_scalar = [
    # Float arrays and flaot scalars.
    (('float16',), float, 'float16'),
    (('float32',), float, 'float32'),
    (('float64',), float, 'float64'),
    (('float64',), float, 'float64'),
    (('float16',), numpy.float64, 'float16'),
    (('float64',), numpy.float16, 'float64'),
]


_in_out_dtypes_arithmetic_scalar = (
    _in_out_dtypes_array_int_scalar
    + _in_out_dtypes_int_array_float_scalar
    + _in_out_dtypes_float_array_float_scalar)


_in_out_dtypes_inplace_arithmetic_scalar = (
    _in_out_dtypes_array_int_scalar
    + _in_out_dtypes_float_array_float_scalar)


_in_out_dtypes_float_arithmetic_scalar = (
    _in_out_dtypes_int_array_float_scalar
    + _in_out_dtypes_float_array_float_scalar)


_in_out_dtypes_inplace_float_arithmetic_scalar = (
    _in_out_dtypes_float_array_float_scalar)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                1, chainerx.testing.numeric_dtypes)),
        'input': ['random'],
        'is_module': [False],
    })
    # is_module
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                1, chainerx.testing.numeric_dtypes)),
        'input': ['random'],
        'is_module': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                1, chainerx.testing.float_dtypes)),
        'input': [float('inf'), -float('inf'), float('nan')],
        'is_module': [False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestNegative(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        if self.is_module:
            return xp.negative(a)
        else:
            return -a


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(chainerx.DtypeError, TypeError))
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_negative_invalid_bool(xp, device, is_module):
    x = xp.array([True, False], dtype='bool_')
    if is_module:
        xp.negative(x)
    else:
        -x


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_binary,
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.numeric_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [False],
    })
    # Dtype combinations
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_dtypes_arithmetic,
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [False],
    })
    # is_module
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.numeric_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': ['random', float('inf'), -float('inf'), float('nan')],
        'input_rhs': ['random', float('inf'), -float('inf'), float('nan')],
        'is_module': [False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestAdd(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        if self.is_module:
            return xp.add(a, b)
        else:
            return a + b


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_arithmetic_invalid)
def test_add_invalid_dtypes(device, dtypes, is_module):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (2, 3)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        if is_module:
            a + b
        else:
            chainerx.add(a, b)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_inplace_binary,
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.numeric_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # Dtype combinations
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_dtypes_inplace_arithmetic,
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # Special values
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': ['random', float('inf'), -float('inf'), float('nan')],
        'input_rhs': ['random', float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestIAdd(math_utils.InplaceBinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        a += b


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_inplace_arithmetic_invalid)
def test_iadd_invalid_dtypes(device, dtypes):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (2, 3)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        a += b


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [False],
    })
    # Type combinations
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [True, False],
    })
    # is_module
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [True, False],
        'is_scalar_rhs': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_float_arithmetic_scalar,
        'input': [float('inf'), -float('inf'), float('nan')],
        'scalar_value': [
            0, -1, 1, 2, float('inf'), -float('inf'), float('nan')],
        'is_module': [False],
        'is_scalar_rhs': [False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestAddScalar(math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        if self.is_module:
            if self.is_scalar_rhs:
                return a + scalar
            else:
                return scalar + a
        else:
            if self.is_scalar_rhs:
                return xp.add(a, scalar)
            else:
                return xp.add(scalar, a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_inplace_arithmetic_scalar,
        'input': ['random'],
        'scalar_value': [1],
    })
    # Dtype combinations
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_inplace_arithmetic_scalar,
        'input': ['random'],
        'scalar_value': [1],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_inplace_float_arithmetic_scalar,
        'input': [float('inf'), -float('inf'), float('nan')],
        'scalar_value': [
            0, -1, 1, 2, float('inf'), -float('inf'), float('nan')],
    })
))
class TestIAddScalar(
        math_utils.InplaceMathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        a += scalar


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_binary,
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.numeric_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [False],
    })
    # Dtype combinations
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_dtypes_arithmetic,
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [False],
    })
    # is_module
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.numeric_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': ['random', float('inf'), -float('inf'), float('nan')],
        'input_rhs': ['random', float('inf'), -float('inf'), float('nan')],
        'is_module': [False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestSub(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        if self.is_module:
            return xp.subtract(a, b)
        else:
            return a - b


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_arithmetic_invalid)
def test_sub_invalid_dtypes(device, dtypes, is_module):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (2, 3)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        if is_module:
            a - b
        else:
            chainerx.subtract(a, b)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_inplace_binary,
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.numeric_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # Dtype combinations
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_dtypes_inplace_arithmetic,
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # Special values
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': ['random', float('inf'), -float('inf'), float('nan')],
        'input_rhs': ['random', float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestISub(math_utils.InplaceBinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        a -= b


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_inplace_arithmetic_invalid)
def test_isub_invalid_dtypes(device, dtypes):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (2, 3)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        a -= b


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [False],
    })
    # Type combinations
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [True, False],
    })
    # is_module
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [True, False],
        'is_scalar_rhs': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_float_arithmetic_scalar,
        'input': [float('inf'), -float('inf'), float('nan')],
        'scalar_value': [
            0, -1, 1, 2, float('inf'), -float('inf'), float('nan')],
        'is_module': [False],
        'is_scalar_rhs': [False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestSubScalar(math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        if self.is_module:
            if self.is_scalar_rhs:
                return a - scalar
            else:
                return scalar - a
        else:
            if self.is_scalar_rhs:
                return xp.subtract(a, scalar)
            else:
                return xp.subtract(scalar, a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_inplace_arithmetic_scalar,
        'input': ['random'],
        'scalar_value': [1],
    })
    # Dtype combinations
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_inplace_arithmetic_scalar,
        'input': ['random'],
        'scalar_value': [1],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_inplace_float_arithmetic_scalar,
        'input': [float('inf'), -float('inf'), float('nan')],
        'scalar_value': [
            0, -1, 1, 2, float('inf'), -float('inf'), float('nan')],
    })
))
class TestISubScalar(
        math_utils.InplaceMathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        a -= scalar


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_binary,
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.all_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [False],
    })
    # Dtype combinations
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': dtype_utils.result_dtypes_two_arrays,
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [False],
    })
    # is_module
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.all_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': ['random', float('inf'), -float('inf'), float('nan')],
        'input_rhs': ['random', float('inf'), -float('inf'), float('nan')],
        'is_module': [False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestMul(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        if self.is_module:
            return xp.multiply(a, b)
        else:
            return a * b


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_inplace_binary,
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.all_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # Dtype combinations
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_dtypes_inplace_arithmetic + [
            ((t, 'bool_'), t) for t in chainerx.testing.all_dtypes
        ],
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # Special values
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': ['random', float('inf'), -float('inf'), float('nan')],
        'input_rhs': ['random', float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestIMul(math_utils.InplaceBinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        a *= b


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [False],
    })
    # Type combinations
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar + [
            ((t,), bool, t) for t in chainerx.testing.all_dtypes
        ],
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [True, False],
    })
    # is_module
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [True, False],
        'is_scalar_rhs': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_float_arithmetic_scalar,
        'input': [float('inf'), -float('inf'), float('nan')],
        'scalar_value': [
            0, -1, 1, 2, float('inf'), -float('inf'), float('nan')],
        'is_module': [False],
        'is_scalar_rhs': [False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestMulScalar(math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        if self.is_module:
            if self.is_scalar_rhs:
                return a * scalar
            else:
                return scalar * a
        else:
            if self.is_scalar_rhs:
                return xp.multiply(a, scalar)
            else:
                return xp.multiply(scalar, a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_inplace_arithmetic_scalar,
        'input': ['random'],
        'scalar_value': [1],
    })
    # Dtype combinations
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': (
            _in_out_dtypes_inplace_arithmetic_scalar + [
                ((t,), bool, t) for t in chainerx.testing.all_dtypes
            ]),
        'input': ['random'],
        'scalar_value': [1],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_inplace_float_arithmetic_scalar,
        'input': [float('inf'), -float('inf'), float('nan')],
        'scalar_value': [
            0, -1, 1, 2, float('inf'), -float('inf'), float('nan')],
    })
))
class TestIMulScalar(
        math_utils.InplaceMathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        a *= scalar


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*chainer.testing.product({
    'lhs,rhs': [
        ([], []),
        ([0, 1, 2, 3, 100, 101, 102, 103], [3] * 8),
        ([-0, -1, -2, -3, -4, -100, -101, -102, -103], [3] * 9),
        ([0, 1, 2, 3, 100, 101, 102, 103], [-3] * 8),
        ([-0, -1, -2, -3, -4, -100, -101, -102, -103], [-3] * 9),
        ([0., 0.8, 1.6, 2.4, 100., 100.8, 101.6, 102.4], [1.2] * 8),
        ([-0., -0.8, -1.6, -2.4, -3.2, -100., -100.8, -101.6, -102.4],
         [1.2] * 9),
        ([0., 0.8, 1.6, 2.4, 100., 100.8, 101.6, 102.4], [-1.2] * 8),
        ([-0., -0.8, -1.6, -2.4, -3.2, -100., -100.8, -101.6, -102.4],
         [-1.2] * 9),
    ],
    'in_dtypes,out_dtype': _in_out_dtypes_arithmetic,
    'is_module': [True, False],
}))
class TestFloorDivide(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    skip_backward_test = True
    skip_double_backward_test = True

    def generate_inputs(self):
        in_dtype1, in_dtype2 = self.in_dtypes
        a = numpy.array(self.lhs).astype(in_dtype1)
        b = numpy.array(self.rhs).astype(in_dtype2)
        return a, b

    def func(self, xp, a, b):
        if self.is_module:
            return xp.floor_divide(a, b)
        else:
            return a // b


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(chainer.testing.product_dict(
    chainer.testing.product({
        'array': [
            ([]),
            ([0, 1, 2, 3, 100, 101, 102, 103]),
            ([-0, -1, -2, -3, -4, -100, -101, -102, -103]),
            ([0., 0.8, 1.6, 2.4, 100., 100.8, 101.6, 102.4]),
            ([-0., -0.8, -1.6, -2.4, -3.2, -100., -100.8, -101.6, -102.4]),
            ([-0.61, -0.6, -0.59, 0.59, 0.6, 0.61]),
        ],
        'is_module': [True, False],
        'is_scalar_rhs': [True, False],
    }),
    chainer.testing.product({
        'scalar_value': [-3, 3, -1.2, 1.2, 0],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
    })
    # Special values
    + chainer.testing.product({
        'scalar_value': [float('inf'), -float('inf'), float('nan')],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_float_arithmetic_scalar,
    })
)))
class TestFloorDivideScalar(
        math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    skip_backward_test = True
    skip_double_backward_test = True

    def setup(self):
        super().setup()
        in_dtype, = self.in_dtypes

        # TODO(imanishi): Remove this.
        if in_dtype == 'uint8' and self.scalar_value < 0:
            self.skip_forward_test = True

    def generate_inputs(self):
        in_dtype, = self.in_dtypes
        a = numpy.array(self.array).astype(in_dtype)
        return a,

    def func_scalar(self, xp, a, scalar):
        if self.is_module:
            if self.is_scalar_rhs:
                return xp.floor_divide(a, scalar)
            else:
                return xp.floor_divide(scalar, a)
        else:
            if self.is_scalar_rhs:
                return a // scalar
            else:
                return scalar // a


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_arithmetic_invalid)
def test_floordiv_invalid_dtypes(device, dtypes, is_module):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (2, 3)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        if is_module:
            a // b
        else:
            chainerx.floor_divide(a, b)


# TODO(imanishi): Support and test zero division and mixed dtypes.
# TODO(imanishi): Support and test chainerx.Scalar // chainerx.ndarray.
# TODO(imanishi): Support and test bool dtype.
@chainerx.testing.numpy_chainerx_array_equal(float16_rtol=1e-3)
@pytest.mark.parametrize('lhs,rhs', [
    ([], []),
    ([0, 1, 2, 3, 100, 101, 102, 103], [3] * 8),
    ([-1, -2, -3, -4, -100, -101, -102, -103], [3] * 8),
    ([0, 1, 2, 3, 100, 101, 102, 103], [-3] * 8),
    ([-1, -2, -3, -4, -100, -101, -102, -103], [-3] * 8),
    ([0., 0.8, 1.6, 2.4, 100., 100.8, 101.6, 102.4], [1.2] * 8),
    ([-0.8, -1.6, -2.4, -3.2, -100., -100.8, -101.6, -102.4], [1.2] * 8),
    ([0., 0.8, 1.6, 2.4, 100., 100.8, 101.6, 102.4], [-1.2] * 8),
    ([-0.8, -1.6, -2.4, -3.2, -100., -100.8, -101.6, -102.4], [-1.2] * 8),
    ([0, 1, 2, 3, 100, 101, 102, 103], 3),
    ([-1, -2, -3, -4, -100, -101, -102, -103], 3),
    ([0, 1, 2, 3, 100, 101, 102, 103], -3),
    ([-1, -2, -3, -4, -100, -101, -102, -103], -3),
    ([0., 0.8, 1.6, 2.4, 100., 100.8, 101.6, 102.4], 1.2),
    ([-0.8, -1.6, -2.4, -3.2, -100., -100.8, -101.6, -102.4], 1.2),
    ([0., 0.8, 1.6, 2.4, 100., 100.8, 101.6, 102.4], -1.2),
    ([-0.8, -1.6, -2.4, -3.2, -100., -100.8, -101.6, -102.4], -1.2),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_ifloordiv(xp, lhs, rhs, device, numeric_dtype):
    if numpy.array(lhs).dtype.kind != numpy.dtype(numeric_dtype).kind:
        return chainerx.testing.ignore()
    lhs = xp.array(lhs).astype(numeric_dtype)
    if isinstance(rhs, (list, tuple)):
        rhs = xp.array(rhs).astype(numeric_dtype)

    lhs //= rhs
    return lhs


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_inplace_arithmetic_invalid)
def test_ifloordiv_invalid_dtypes(device, dtypes):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (2, 3)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        a //= b


_in_out_dtypes_inplace_truediv = [
    (('float32', 'int16'), 'float32'),
    (('float64', 'uint8'), 'float64'),
    (('float16', 'float16'), 'float16'),
    (('float32', 'float32'), 'float32'),
    (('float64', 'float64'), 'float64'),
    (('float32', 'float16'), 'float32'),
    (('float16', 'float64'), 'float64'),
]


_in_out_dtypes_truediv = _in_out_dtypes_inplace_truediv + [
    (('int8', 'int8'), 'float32'),
    (('int16', 'int16'), 'float32'),
    (('int32', 'int32'), 'float32'),
    (('int64', 'int64'), 'float32'),
    (('uint8', 'uint8'), 'float32'),
    (('int8', 'int32'), 'float32'),
    (('uint8', 'int64'), 'float32'),
    (('int8', 'uint8'), 'float32'),
    (('int32', 'float16'), 'float16'),
    (('uint8', 'float32'), 'float32'),
]


_in_out_dtypes_inplace_truediv_scalar = [
    (('int8',), int, 'float32'),
    (('int16',), int, 'float32'),
    (('int32',), int, 'float32'),
    (('int64',), int, 'float32'),
    (('uint8',), int, 'float32'),
    (('float16',), int, 'float16'),
    (('float32',), int, 'float32'),
    (('float64',), int, 'float64'),
    (('float16',), float, 'float16'),
    (('float32',), float, 'float32'),
    (('float64',), float, 'float64'),
]


_in_out_dtypes_truediv_scalar = _in_out_dtypes_inplace_truediv_scalar + [
    (('int8',), float, 'float32'),
    (('int16',), float, 'float32'),
    (('int32',), float, 'float32'),
    (('int64',), float, 'float32'),
    (('uint8',), float, 'float32'),
]


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_binary,
        'in_dtypes,out_dtype': _in_out_dtypes_truediv,
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [False],
    })
    # Dtype combinations
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_dtypes_truediv,
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [False],
    })
    # is_module
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_dtypes_truediv,
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': ['random', float('inf'), -float('inf'), float('nan')],
        'input_rhs': ['random', float('inf'), -float('inf'), float('nan')],
        'is_module': [False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestTrueDivide(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    check_numpy_strides_compliance = False

    def setup(self):
        super().setup()
        dtype1, dtype2 = self.in_dtypes
        if dtype1 == 'float16' or dtype2 == 'float16':
            self.check_forward_options.update({'rtol': 5e-3, 'atol': 5e-3})
            self.check_backward_options.update({'rtol': 5e-3, 'atol': 5e-3})
            self.check_double_backward_options.update(
                {'rtol': 5e-3, 'atol': 5e-3})

    def generate_inputs(self):
        a, b = super().generate_inputs()
        if self.input_lhs == 'random':
            # Avoid (-0.3, 0.3) interval
            with math_utils.IgnoreNumpyFloatingPointError():
                b[numpy.logical_and(-0.3 < b, b < 0.3)] = 1
        return a, b

    def func(self, xp, a, b):
        if self.is_module:
            return xp.divide(a, b)
        else:
            return a / b


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_arithmetic_invalid)
def test_truediv_invalid_dtypes(device, dtypes, is_module):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (2, 3)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        if is_module:
            a / b
        else:
            chainerx.true_divide(a, b)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_inplace_binary,
        'in_dtypes,out_dtype': _in_out_dtypes_inplace_truediv,
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # Dtype combinations
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_dtypes_inplace_truediv,
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # Special values
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': ['random', float('inf'), -float('inf'), float('nan')],
        'input_rhs': ['random', float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestITrueDivide(
        math_utils.InplaceBinaryMathTestBase, op_utils.NumpyOpTest):

    skip_backward_test = True
    skip_double_backward_test = True

    def generate_inputs(self):
        a, b = super().generate_inputs()
        if self.input_lhs == 'random':
            with math_utils.IgnoreNumpyFloatingPointError():
                b[numpy.logical_and(-0.3 < b, b < 0.3)] = 1
        return a, b

    def func(self, xp, a, b):
        a /= b


# TODO(hvy): Support and test zero division and mixed dtypes (dtype kinds).
@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_truediv_scalar,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [True, False],
    })
    # Dtype combinations
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_truediv_scalar,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                1, chainerx.testing.float_dtypes)),
        'scalar_type': [float],
        'input': [float('inf'), -float('inf'), float('nan')],
        'scalar_value': [-1, 1, 2, float('inf'), -float('inf'), float('nan')],
        'is_module': [False],
        'is_scalar_rhs': [True, False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestTrueDivideScalar(
        math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    check_numpy_strides_compliance = False

    def generate_inputs(self):
        # Do not divide by small number to avoid ridiculously large outputs.
        if not self.is_scalar_rhs and self.input == 'random':
            in_dtype, = self.in_dtypes
            low = -5 if numpy.dtype(in_dtype).kind != 'u' else 2
            high = 5
            x = array_utils.uniform(self.shape, in_dtype, low=low, high=high)
            x[(-1 < x) & (x < 0)] = -2
            x[(0 <= x) & (x < 1)] = 2
            return x,
        return super().generate_inputs()

    def func_scalar(self, xp, a, scalar):
        if self.is_module:
            if self.is_scalar_rhs:
                return xp.divide(a, scalar)
            else:
                return xp.divide(scalar, a)
        else:
            if self.is_scalar_rhs:
                return a / scalar
            else:
                return scalar / a


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                1, chainerx.testing.float_dtypes)),
        'scalar_type': [float],
        'input': ['random'],
        'scalar_value': [1],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                1, chainerx.testing.float_dtypes)),
        'scalar_type': [float],
        'input': [float('inf'), -float('inf'), float('nan')],
        'scalar_value': [-1, 1, 2, float('inf'), -float('inf'), float('nan')],
    })
))
class TestITrueDivideScalar(
        math_utils.InplaceMathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        a /= scalar


_in_out_dtypes_bitwise = dtype_utils._permutate_dtype_mapping([
    # Same dtypes
    (('bool_', 'bool_'), 'bool_'),
    (('int8', 'int8'), 'int8'),
    (('int16', 'int16'), 'int16'),
    (('int32', 'int32'), 'int32'),
    (('int64', 'int64'), 'int64'),
    (('uint8', 'uint8'), 'uint8'),
    # Mixed dtypes
    (('bool_', 'int8'), 'int8'),
    (('bool_', 'int16'), 'int16'),
    (('bool_', 'int32'), 'int32'),
    (('bool_', 'int64'), 'int64'),
    (('bool_', 'uint8'), 'uint8'),
    (('int8', 'int16'), 'int16'),
    (('int8', 'int32'), 'int32'),
    (('int8', 'int64'), 'int64'),
    (('int8', 'uint8'), 'int16'),
    (('int16', 'int32'), 'int32'),
    (('int16', 'int64'), 'int64'),
    (('int16', 'uint8'), 'int16'),
    (('int32', 'int64'), 'int64'),
    (('int32', 'uint8'), 'int32'),
    (('int64', 'uint8'), 'int64'),
])

_inplace_invalid_bitwise = [
    (('bool_', 'int8'), 'int8'),
    (('bool_', 'int16'), 'int16'),
    (('bool_', 'int32'), 'int32'),
    (('bool_', 'int64'), 'int64'),
    (('bool_', 'uint8'), 'uint8'),
]

_in_out_inplace_dtypes_bitwise = [
    dtypes for dtypes in _in_out_dtypes_bitwise
    if dtypes not in _inplace_invalid_bitwise
]

_scalar_invalid_bitwise = [
    (('float16',), int, 'float16'),
    (('float32',), int, 'float32'),
    (('float64',), int, 'float64'),
    (('float64',), numpy.int8, 'float64'),
    (('float16',), numpy.int64, 'float16'),
]

_in_out_scalar_dtypes_bitwise = [
    dtypes for dtypes in _in_out_dtypes_array_int_scalar
    if dtypes not in _scalar_invalid_bitwise
]

_params_bitwise = (
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_binary,
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.nonfloat_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [False],
    })
    # Dtype combinations
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_dtypes_bitwise,
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [False],
    })
    # is_module
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.nonfloat_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [True, False],
    })
)


_inplace_params_bitwise = (
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_inplace_binary,
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.nonfloat_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # Dtype combinations
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_inplace_dtypes_bitwise,
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # Special values
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.nonfloat_dtypes)),
        'input_lhs': ['random', float('inf'), -float('inf'), float('nan')],
        'input_rhs': ['random', float('inf'), -float('inf'), float('nan')],
    })
)


_scalar_params_bitwise = (
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_scalar_dtypes_bitwise,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [False],
    })
    # Type combinations
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_scalar_dtypes_bitwise,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [True, False],
    })
    # is_module
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_scalar_dtypes_bitwise,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [True, False],
        'is_scalar_rhs': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_scalar_dtypes_bitwise,
        'input': [float('inf'), -float('inf'), float('nan')],
        'scalar_value': [
            0, -1, 1, 2],
        'is_module': [False],
        'is_scalar_rhs': [False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
)


_inplace_scalar_params_bitwise = (
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_scalar_dtypes_bitwise,
        'input': ['random'],
        'scalar_value': [1],
    })
    # Dtype combinations
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_scalar_dtypes_bitwise,
        'input': ['random'],
        'scalar_value': [1],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_scalar_dtypes_bitwise,
        'input': [float('inf'), -float('inf'), float('nan')],
        'scalar_value': [
            0, -1, 1, 2],
    })
)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_params_bitwise)
class TestBitwiseAnd(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        if self.is_module:
            return xp.bitwise_and(a, b)
        else:
            return a & b


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_params_bitwise)
class TestBitwiseOr(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        if self.is_module:
            return xp.bitwise_or(a, b)
        else:
            return a | b


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_params_bitwise)
class TestBitwiseXor(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        if self.is_module:
            return xp.bitwise_xor(a, b)
        else:
            return a ^ b


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_inplace_params_bitwise)
class TestIBitwiseAnd(
        math_utils.InplaceBinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        a &= b


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_inplace_params_bitwise)
class TestIBitwiseOr(
        math_utils.InplaceBinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        a |= b


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_inplace_params_bitwise)
class TestIBitwiseXor(
        math_utils.InplaceBinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        a ^= b


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _inplace_invalid_bitwise)
def test_iand_invalid_dtypes(device, dtypes):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (2, 3)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        a &= b


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _inplace_invalid_bitwise)
def test_ior_invalid_dtypes(device, dtypes):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (2, 3)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        a |= b


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _inplace_invalid_bitwise)
def test_ixor_invalid_dtypes(device, dtypes):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (2, 3)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        a ^= b


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_scalar_params_bitwise)
class TestBitwiseAndScalar(
        math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        if self.is_module:
            if self.is_scalar_rhs:
                return a & scalar
            else:
                return scalar & a
        else:
            if self.is_scalar_rhs:
                return xp.bitwise_and(a, scalar)
            else:
                return xp.bitwise_and(scalar, a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_scalar_params_bitwise)
class TestBitwiseOrScalar(math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        if self.is_module:
            if self.is_scalar_rhs:
                return a | scalar
            else:
                return scalar | a
        else:
            if self.is_scalar_rhs:
                return xp.bitwise_or(a, scalar)
            else:
                return xp.bitwise_or(scalar, a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_scalar_params_bitwise)
class TestBitwiseXorScalar(
        math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        if self.is_module:
            if self.is_scalar_rhs:
                return a ^ scalar
            else:
                return scalar ^ a
        else:
            if self.is_scalar_rhs:
                return xp.bitwise_xor(a, scalar)
            else:
                return xp.bitwise_xor(scalar, a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_inplace_scalar_params_bitwise)
class TestIBitwiseAndScalar(
        math_utils.InplaceMathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        a &= scalar


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_inplace_scalar_params_bitwise)
class TestIBitwiseOrScalar(
        math_utils.InplaceMathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        a |= scalar


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_inplace_scalar_params_bitwise)
class TestIBitwiseXorScalar(
        math_utils.InplaceMathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        a ^= scalar


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('in_dtypes,out_dtype', [
    (('bool_',), 'int64'),
    (('int8',), 'int64'),
    (('int16',), 'int64'),
    (('int32',), 'int64'),
    (('int64',), 'int64'),
    (('float16',), 'float16'),
    (('float32',), 'float32'),
    (('float64',), 'float64'),

    # TODO(niboshi): Unsigned integer dtypes should result in uint64.
    # Currently chainerx returns int64.
    (('uint8',), 'int64'),
])
@chainer.testing.parameterize_pytest('shape,axis', [
    ((), None),
    ((), ()),
    ((2,), None),
    ((2,), ()),
    ((2,), 0),
    ((2,), (0,)),
    ((2,), (-1,)),
    ((2, 3), None),
    ((2, 3), ()),
    ((2, 3), 0),
    ((2, 3), (0,)),
    ((2, 3), (1,)),
    ((2, 3), (-1,)),
    ((2, 3), (-2,)),
    ((2, 3), (0, 1)),
    ((2, 3), (-2, -1)),
    ((1, 3), None),  # sum over 1-dim axis
    ((0, 3), None),  # sum over 0-dim axis
    # Sum over axes that are in the middle or apart
    ((2, 3, 4), (1,)),
    ((2, 3, 4), (0, 2)),
    # Sum over axes that are apart and/or unsorted
    ((2, 3), (1, 0)),
    ((2, 3, 4), (2, 0)),
    ((2, 3, 4), (2, 0, 1)),
    ((2, 3, 4), (-2, 2, 0)),
])
@chainer.testing.parameterize_pytest('keepdims', [True, False])
@chainer.testing.parameterize_pytest('is_module', [True, False])
class TestSum(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    input = 'random'

    def setup(self):
        super().setup()
        in_dtype, = self.in_dtypes
        if in_dtype == 'float16':
            self.check_forward_options.update({'rtol': 1e-2, 'atol': 1e-2})
            self.check_backward_options.update({'rtol': 1e-2, 'atol': 1e-2})
            self.check_double_backward_options.update(
                {'rtol': 1e-2, 'atol': 1e-2})

    def func(self, xp, a):
        if self.is_module:
            return xp.sum(a, axis=self.axis, keepdims=self.keepdims)
        else:
            return a.sum(axis=self.axis, keepdims=self.keepdims)


@op_utils.op_test(['native:0'])
class TestSumStability(op_utils.NumpyOpTest):

    skip_backward_test = True
    skip_double_backward_test = True

    def generate_inputs(self):
        return numpy.full(2 ** 20, 0.1, dtype=numpy.float32),

    def forward_xp(self, inputs, xp):
        x, = inputs
        if xp is chainerx:
            return x.sum(),
        else:
            return (x[0] * x.size).astype(x.dtype),


@op_utils.op_test(['native:0'])
@chainer.testing.parameterize_pytest('size', list(range(1024)))
class TestSumEachSize(op_utils.NumpyOpTest):

    skip_backward_test = True
    skip_double_backward_test = True

    def generate_inputs(self):
        return numpy.arange(self.size, dtype=numpy.int32) + 1,

    def forward_xp(self, inputs, xp):
        x, = inputs
        return x.sum(),


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(chainerx.DimensionError, ValueError))
@pytest.mark.parametrize('keepdims', [False, True])
@pytest.mark.parametrize('shape,axis', [
    # ((), 0), # TODO(sonots): Fix compatibility
    ((), 1),
    ((), (1,)),
    ((2,), 2),
    ((2,), (2,)),
    ((2,), (-2,)),
    ((2, 3,), (-3,)),
    ((2, 3,), (-3, -4)),
    ((2, 3,), (0, 0)),
    ((2, 3,), (-1, -1)),
    ((2, 3,), (0, 1, 1)),
    ((2, 3,), (0, -2)),
])
def test_sum_invalid(is_module, xp, shape, axis, keepdims, dtype):
    a = array_utils.create_dummy_ndarray(xp, shape, dtype)
    if is_module:
        xp.sum(a, axis=axis, keepdims=keepdims)
    else:
        a.sum(axis=axis, keepdims=keepdims)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': ['random'],
        'scalar_value': [1],
        'is_scalar_rhs': [False],
    })
    # Differentiable cases
    + chainer.testing.product({
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': [numpy.array([1, 3, 3, 4])],
        'scalar_value': [0, 2, 5],
        'is_scalar_rhs': [False, True],
    })
    # Non-differentiable cases
    + chainer.testing.product({
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': [numpy.array([1, 3, 3, 4])],
        'scalar_value': [1, 3, 4],
        'is_scalar_rhs': [False, True],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
    # Special float values
    + chainer.testing.product({
        'in_dtypes,scalar_type,out_dtype': (
            _in_out_dtypes_float_arithmetic_scalar),
        # TODO(imanishi): Add test for NaN.
        'input': [numpy.array([0, float('inf'), -float('inf')])],
        'scalar_value': [-1, 0, 1, float('inf'), -float('inf')],
        'is_scalar_rhs': [False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestMinimumScalar(math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    dodge_nondifferentiable = True

    def func_scalar(self, xp, a, scalar):
        if self.is_scalar_rhs:
            return xp.minimum(a, scalar)
        else:
            return xp.minimum(scalar, a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': ['random'],
        'scalar_value': [0, 1],
        'is_scalar_rhs': [False],
    })
    # Differentiable cases
    + chainer.testing.product({
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': [numpy.array([1, 3, 3, 4])],
        'scalar_value': [0, 2, 5],
        'is_scalar_rhs': [False, True],
    })
    # Non-differentiable cases
    + chainer.testing.product({
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': [numpy.array([1, 3, 3, 4])],
        'scalar_value': [1, 3, 4],
        'is_scalar_rhs': [False, True],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
    # Special float values
    + chainer.testing.product({
        'in_dtypes,scalar_type,out_dtype': (
            _in_out_dtypes_float_arithmetic_scalar),
        # TODO(imanishi): Add test for NaN.
        'input': [numpy.array([0, float('inf'), -float('inf')])],
        'scalar_value': [-1, 0, 1, float('inf'), -float('inf')],
        'is_scalar_rhs': [False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestMaximumScalar(math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    dodge_nondifferentiable = True

    def func_scalar(self, xp, a, scalar):
        if self.is_scalar_rhs:
            return xp.maximum(a, scalar)
        else:
            return xp.maximum(scalar, a)


def _create_dummy_array_for_dot(xp, shape, dtype):
    x = numpy.arange(numpy.prod(shape)).reshape(shape)
    if dtype == 'bool_':
        x = numpy.asarray(x % 2 == 0)
    else:
        x = x.astype(dtype)
    return xp.array(x)


# An association list that associates a dtype to the type which ChainerX's
# real-valued functions should return.
_in_out_float_dtypes_math_functions = [
    # Float.
    (('float16',), 'float16'),
    (('float32',), 'float32'),
    (('float64',), 'float64'),
]


_in_out_dtypes_math_functions = _in_out_float_dtypes_math_functions + [
    # Signed int.
    (('int8',), 'float32'),
    (('int16',), 'float32'),
    (('int32',), 'float32'),
    (('int64',), 'float32'),
    # Unsigned int.
    (('uint8',), 'float32'),
    # Bool.
    (('bool_',), 'float32'),
]


_in_out_dtypes_math_binary_functions = dtype_utils._permutate_dtype_mapping([
    # integer mixed
    (('int8', 'int16'), 'float32'),
    (('int8', 'int32'), 'float32'),
    (('int8', 'int64'), 'float32'),
    (('int8', 'uint8'), 'float32'),
    (('int16', 'int32'), 'float32'),
    (('int16', 'int64'), 'float32'),
    (('int16', 'uint8'), 'float32'),
    (('int32', 'int64'), 'float32'),
    (('int32', 'uint8'), 'float32'),
    (('int64', 'uint8'), 'float32'),
    # integer float mixed
    (('int8', 'float16'), 'float16'),
    (('int8', 'float32'), 'float32'),
    (('int8', 'float64'), 'float64'),
    (('int16', 'float16'), 'float16'),
    (('int16', 'float32'), 'float32'),
    (('int16', 'float64'), 'float64'),
    (('int32', 'float16'), 'float16'),
    (('int32', 'float32'), 'float32'),
    (('int32', 'float64'), 'float64'),
    (('int64', 'float16'), 'float16'),
    (('int64', 'float32'), 'float32'),
    (('int64', 'float64'), 'float64'),
    (('uint8', 'float16'), 'float16'),
    (('uint8', 'float32'), 'float32'),
    (('uint8', 'float64'), 'float64'),
    # float mixed
    (('float16', 'float32'), 'float32'),
    (('float16', 'float64'), 'float64'),
    (('float32', 'float64'), 'float64'),
])


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (1,), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
        'input': [0, 2, -2],
    })
    # Special shapes (array.size = 0)
    + chainer.testing.product({
        'shape': [(0), (2, 0, 3)],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
        'input': [0, 2, -2],
        'check_numpy_strides_compliance': [False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
        'input': [float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestExp(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.exp(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (1,), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
        'input': [1, 3],
    })
    # Special shapes (array.size = 0)
    + chainer.testing.product({
        'shape': [(0,), (2, 0, 3)],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
        'input': [1, 3],
        'check_numpy_strides_compliance': [False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
        'input': [float('inf'), -float('inf'), float('nan'), -1, 0],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestLog(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.log(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (1,), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
        'input': [1, 3],
    })
    # Special shapes (array.size = 0)
    + chainer.testing.product({
        'shape': [(0,), (2, 0, 3)],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
        'input': [1, 3],
        'check_numpy_strides_compliance': [False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
        'input': [float('inf'), -float('inf'), float('nan'), -1, 0],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestLog10(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.log10(a)


_logsumexp_params = [
    ((2,), 0),
    ((2,), -1),
    ((2, 3), None),
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 3), -2),
    ((2, 3), (0, 1)),
    ((2, 3), (-2, 1)),
    ((1, 2, 3), None),
    ((1, 2, 3), (1)),
    ((1, 2, 3), (1, 0)),
    ((1, 2, 3), (0, 1, 2)),
]


_invalid_logsumexp_params = [
    # Axis out of bounds
    ((2,), 1),
    ((2,), -2),
    ((2,), (0, 1)),
    ((2, 3), (0, 1, 2)),
    # Duplicate axes
    ((2,), (0, 0)),
    ((2, 3), (0, 0)),
]


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest(
    'in_dtypes,out_dtype', _in_out_dtypes_math_functions)
@chainer.testing.parameterize_pytest('shape,axis', _logsumexp_params)
@chainer.testing.parameterize_pytest('keepdims', [True, False])
class TestLogSumExp(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    input = 'random'

    def setup(self):
        super().setup()
        if self.in_dtypes == 'float16':
            # TODO(imanishi): Support device implementation and remove this.
            self.check_forward_options.update({'rtol': 3e-3, 'atol': 3e-3})

    def forward_xp(self, inputs, xp):
        x, = inputs
        axis = self.axis
        keepdims = self.keepdims
        if xp is chainerx:
            return chainerx.logsumexp(x, axis=axis, keepdims=keepdims),
        x = x.astype(self.out_dtype)
        return numpy.log(numpy.exp(x).sum(axis=axis, keepdims=keepdims)),


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _invalid_logsumexp_params)
@pytest.mark.parametrize('keepdims', [True, False])
# TODO(hvy): Should not overflow for large numbers, add tests
def test_logsumexp_invalid(device, a_shape, axis, keepdims, dtype):
    a = array_utils.create_dummy_ndarray(chainerx, a_shape, dtype)
    with pytest.raises(chainerx.DimensionError):
        chainerx.logsumexp(a, axis=axis, keepdims=keepdims)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shape,axis', _logsumexp_params)
@chainer.testing.parameterize_pytest(
    'in_dtypes,out_dtype', _in_out_dtypes_math_functions)
class TestLogSoftmax(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    input = 'random'

    def setup(self):
        super().setup()
        self.check_forward_options.update({'rtol': 3e-3, 'atol': 3e-3})
        self.check_backward_options.update({'rtol': 3e-3, 'atol': 3e-3})

    def forward_xp(self, inputs, xp):
        x, = inputs
        axis = self.axis
        if xp is chainerx:
            return chainerx.log_softmax(x, axis=axis),
        x = x.astype(self.out_dtype)
        axis = axis if axis is not None else 1
        return x - numpy.log(numpy.exp(x).sum(axis=axis, keepdims=True)),


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _invalid_logsumexp_params)
def test_log_softmax_invalid(device, a_shape, axis, dtype):
    a = array_utils.create_dummy_ndarray(chainerx, a_shape, dtype)
    with pytest.raises(chainerx.DimensionError):
        return chainerx.log_softmax(a, axis=axis)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': ['random', float('inf'), -float('inf'), float('nan')],
        'input_rhs': ['random', float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestSquaredDifference(op_utils.OpTest):

    def setup(self):
        x1_dtype, x2_dtype = self.in_dtypes

        if x1_dtype == 'float16' or x2_dtype == 'float16':
            self.check_forward_options.update({'atol': 3e-3, 'rtol': 3e-3})
            self.check_backward_options.update({'atol': 1e-2, 'rtol': 5e-2})
            self.check_double_backward_options.update(
                {'atol': 1e-2, 'rtol': 5e-2})

    def generate_inputs(self):
        shape = self.shape
        x1_dtype, x2_dtype = self.in_dtypes
        x1 = array_utils.uniform(shape, x1_dtype)
        x2 = array_utils.uniform(shape, x2_dtype)
        return x1, x2

    def forward_chainerx(self, inputs):
        x1, x2 = inputs
        y = chainerx.squared_difference(x1, x2)
        return y,

    def forward_expected(self, inputs):
        x1, x2 = inputs
        y = numpy.asarray(
            numpy.square(numpy.subtract(x1, x2))).astype(x1.dtype)
        return y,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Differentiable
    chainer.testing.product({
        'input': [
            numpy.asarray(0.),
            numpy.asarray(-1.),
            numpy.asarray(1.),
            numpy.asarray(10.),
            numpy.full((), 2.),
            numpy.full((0,), 2.),
            numpy.full((2, 3), 2.)
        ]})
    +
    # Nondifferentiable
    chainer.testing.product({
        'input': [
            numpy.asarray(float('inf')),
            numpy.asarray(float('nan')),
        ],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
@pytest.mark.parametrize('contiguous', [None, 'C'])
class TestSigmoid(op_utils.NumpyOpTest):

    def setup(self, contiguous, float_dtype):
        self.dtype = float_dtype
        self.contiguous = contiguous
        self.check_forward_options = {'atol': 5e-3, 'rtol': 5e-3}

        if float_dtype == 'float16':
            self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-2}

    def generate_inputs(self):
        return self.input,

    def forward_xp(self, inputs, xp):
        if xp is numpy:
            return 1 / (1 + numpy.exp(-inputs[0])),
        return xp.sigmoid(inputs[0]),


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shape,axis', _logsumexp_params)
@chainer.testing.parameterize_pytest(
    'in_dtypes,out_dtype', _in_out_dtypes_math_functions)
class TestSoftmax(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    input = 'random'

    def setup(self):
        super().setup()
        self.check_forward_options.update({'rtol': 3e-3, 'atol': 3e-3})
        self.check_backward_options.update({'rtol': 3e-3, 'atol': 3e-3})

    def forward_xp(self, inputs, xp):
        x, = inputs
        axis = self.axis
        if xp is chainerx:
            return chainerx.softmax(x, axis=axis),
        x = x.astype(self.out_dtype)
        axis = axis if axis is not None else 1
        return numpy.exp(x) / (numpy.exp(x).sum(axis=axis, keepdims=True)),


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
        'input': [-2, 0, 2],
        'contiguous': [None, 'C'],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
        'input': [float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestSquare(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.square(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (1,), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
        'input': [1, 3],
    })
    # Special shapes (array.size = 0)
    + chainer.testing.product({
        'shape': [(0,), (2, 0, 3)],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
        'input': [1, 3],
        'check_numpy_strides_compliance': [False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
        'input': [float('inf'), -float('inf'), float('nan'), -1, 0],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestSqrt(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.sqrt(a)


_trigonometric_hyperbolic_params = \
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
        'input': [-2, 0, 2],
        'contiguous': [None, 'C'],
    }) + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
        'input': [1.57, 2, 3.14, float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _trigonometric_hyperbolic_params
))
class TestSinh(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.sinh(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _trigonometric_hyperbolic_params
))
class TestCosh(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.cosh(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _trigonometric_hyperbolic_params
))
class TestTanh(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.tanh(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _trigonometric_hyperbolic_params
))
class TestSin(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.sin(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _trigonometric_hyperbolic_params
))
class TestCos(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.cos(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _trigonometric_hyperbolic_params
))
class TestTan(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    dodge_nondifferentiable = True
    check_backward_options = {'atol': 3e-5}

    def func(self, xp, a):
        return xp.tan(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
        'input': ['random'],
        'contiguous': [None, 'C'],
    })
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
        'input': [float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestAbs(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        assert chainerx.abs is chainerx.absolute
        return xp.abs(a)


def _make_inverse_trig_params(name):
    # Makes test parameters for inverse trigonometric functions

    inverse_trig_differentiable_inputs = {
        'arcsin': [-0.9, 0, 0.9],
        'arccos': [-0.9, 0, 0.9],
        'arctan': [-3, -0.2, 0, 0.2, 3],
        'arcsinh': [-3, -0.2, 0, 0.2, 3],
        'arccosh': [1.2, 3],
        'arctanh': [-0.9, 0, 0.9],
    }

    inverse_trig_nondifferentiable_inputs = {
        'arcsin': [-3, -1, 1, 3],
        'arccos': [-3, -1, 1, 3],
        'arctan': [],
        'arcsinh': [],
        'arccosh': [-3, 0, 0.2, 1],
        'arctanh': [-3, -1, 1, 3],
    }

    nonfinite_numbers = [float('inf'), -float('inf'), float('nan')]

    return (
        # Various shapes and differentiable inputs
        chainer.testing.product({
            'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
            'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
            'input': inverse_trig_differentiable_inputs[name],
            'contiguous': [None, 'C'],
        })
        +
        # Nondifferentiable inputs
        chainer.testing.product({
            'shape': [(2, 3)],
            'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
            'input': (
                inverse_trig_nondifferentiable_inputs[name]
                + nonfinite_numbers),
            'skip_backward_test': [True],
            'skip_double_backward_test': [True],
        }))


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _make_inverse_trig_params('arcsinh')
))
class TestArcsinh(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.arcsinh(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _make_inverse_trig_params('arccosh')
))
class TestArccosh(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.arccosh(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _make_inverse_trig_params('arcsin')
))
class TestArcsin(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.arcsin(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _make_inverse_trig_params('arccos')
))
class TestArccos(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.arccos(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _make_inverse_trig_params('arctan')
))
class TestArctan(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.arctan(a)


# Since the gradient of arctan2 is quite flaky.
# for smaller values especially `float16`.
@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_binary,
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': [1],
        'input_rhs': [2],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
    # Differentiable points
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': [-3, -0.75, 0.75, 3],
        'input_rhs': [-3, -0.75, 0.75, 3],
    })
    # Mixed dtypes
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_dtypes_math_binary_functions,
        'input_lhs': [-1.],
        'input_rhs': [-1.],
    })
    # Special values
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': ['random', float('inf'), -float('inf'), float('nan'),
                      +0.0, -0.0],
        'input_rhs': ['random', float('inf'), -float('inf'), float('nan'),
                      +0.0, -0.0],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestArctan2(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        return xp.arctan2(a, b)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0.5),
    numpy.asarray(-1.2),
    numpy.asarray(10.9),
    numpy.asarray(float('inf')),
    numpy.asarray(-float('inf')),
    numpy.asarray(float('nan')),
    numpy.full((), 2.1),
    numpy.full((0,), 2),
    numpy.full((2, 3), 2.6),
    numpy.full((1, 1), 1.01),
    numpy.full((1, 1), 1.99),
])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_math_functions)
@pytest.mark.parametrize('func', [
    lambda xp, a: xp.ceil(a),
    lambda xp, a: xp.floor(a)
])
def test_rounding_routines(func, xp, device, input, dtypes):
    (in_dtype, ), out_dtype = dtypes
    a = xp.array(input.astype(in_dtype))
    a = func(xp, a)
    a = dtype_utils.cast_if_numpy_array(xp, a, out_dtype)
    return a


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0), numpy.asarray(-1), numpy.asarray(
        10), numpy.asarray(float('inf')), numpy.asarray(-float('inf')),
    numpy.asarray(float('nan')), numpy.full(
        (), 2), numpy.full((0,), 2), numpy.full((2, 3), 2)
])
def test_isnan(xp, device, input, dtype):
    a = xp.array(input.astype(dtype))
    return xp.isnan(a)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0), numpy.asarray(-1), numpy.asarray(
        10), numpy.asarray(float('inf')), numpy.asarray(-float('inf')),
    numpy.asarray(float('nan')), numpy.full(
        (), 2), numpy.full((0,), 2), numpy.full((2, 3), 2)
])
def test_isinf(xp, device, input, dtype):
    a = xp.array(input.astype(dtype))
    return xp.isinf(a)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0), numpy.asarray(-1), numpy.asarray(
        10), numpy.asarray(float('inf')), numpy.asarray(-float('inf')),
    numpy.asarray(float('nan')), numpy.full(
        (), 2), numpy.full((0,), 2), numpy.full((2, 3), 2)
])
def test_isfinite(xp, device, input, dtype):
    a = xp.array(input.astype(dtype))
    return xp.isfinite(a)


def test_max_amax():
    assert chainerx.amax is chainerx.max


_minmax_params = [
    # --- single axis
    # input, axis
    (numpy.asarray(0), None),
    (numpy.asarray(-1), None),
    (numpy.asarray(float('inf')), None),
    (numpy.asarray(float('nan')), None),
    (numpy.asarray(-float('inf')), None),
    (numpy.asarray([4, 1, 4, 1]), None),
    (numpy.asarray([4, 1, 4, 1]), 0),
    (numpy.asarray([[4, 4, 1, 1], [4, 1, 4, 1]]), 0),
    (numpy.asarray([[4, 4, 1, 1], [4, 1, 4, 1]]).T, 1),
    (numpy.asarray([-0.0, +0.0, +0.0, -0.0]), None),
    (numpy.asarray([[True, True, False, False],
                    [True, False, True, False]]), 0),
    (numpy.ones((2, 3)), 1),
    (numpy.ones((2, 3)), -2),
    # --- multiple axes
    # input, axis
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (0, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (-2, -1)),
]


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape,axis': [
            ((), None),
            ((4,), None),
            ((4,), 0),
            ((4, 2), None),
            ((4, 2), 0),
            ((4, 2), 1),
            ((4, 2), -2),
            ((4, 3), (0, 1)),
            ((4, 3), (-2, -1)),
        ],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                1, chainerx.testing.all_dtypes)),
        'is_module': [True, False],
    }) +
    chainer.testing.product({
        'array,axis': _minmax_params,
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                1, chainerx.testing.all_dtypes)),
        'is_module': [True, False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestMax(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    dodge_nondifferentiable = True

    def generate_inputs(self):
        in_dtype, = self.in_dtypes
        if hasattr(self, 'array'):
            return self.array.astype(in_dtype),
        return array_utils.uniform(self.shape, in_dtype),

    def func(self, xp, a):
        if self.is_module:
            return xp.max(a, self.axis)
        else:
            return a.max(self.axis)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('array,axis', [
    (numpy.ones((2, 3)), 2),
    (numpy.ones((2, 3)), -3),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (-3, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 2)),
])
@pytest.mark.parametrize('dtype', chainerx.testing.all_dtypes)
@pytest.mark.parametrize('is_module', [True, False])
def test_max_invalid_shapes_and_axis(device, array, axis, dtype, is_module):
    a = chainerx.array(array).astype(dtype)
    with pytest.raises(chainerx.DimensionError):
        if is_module:
            chainerx.max(a, axis)
        else:
            a.max(axis)


def test_min_amin():
    assert chainerx.amin is chainerx.min


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape,axis': [
            ((), None),
            ((4,), None),
            ((4,), 0),
            ((4, 2), None),
            ((4, 2), 0),
            ((4, 2), 1),
            ((4, 2), -2),
            ((4, 3), (0, 1)),
            ((4, 3), (-2, -1)),
        ],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                1, chainerx.testing.all_dtypes)),
        'is_module': [True, False],
    }) +
    chainer.testing.product({
        'array,axis': _minmax_params,
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                1, chainerx.testing.all_dtypes)),
        'is_module': [True, False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestMin(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    dodge_nondifferentiable = True

    def generate_inputs(self):
        in_dtype, = self.in_dtypes
        if hasattr(self, 'array'):
            return self.array.astype(in_dtype),
        return array_utils.uniform(self.shape, in_dtype),

    def func(self, xp, a):
        if self.is_module:
            return xp.min(a, self.axis)
        else:
            return a.min(self.axis)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('array,axis', [
    (numpy.ones((2, 3)), 2),
    (numpy.ones((2, 3)), -3),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (-3, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 2)),
])
@pytest.mark.parametrize('dtype', chainerx.testing.all_dtypes)
@pytest.mark.parametrize('is_module', [True, False])
def test_min_invalid_shapes_and_axis(device, array, axis, dtype, is_module):
    a = chainerx.array(array).astype(dtype)
    with pytest.raises(chainerx.DimensionError):
        if is_module:
            chainerx.min(a, axis)
        else:
            a.min(axis)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_binary,
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.numeric_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # Dtype combinations
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_dtypes_arithmetic,
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [False],
    })
    # is_module
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.numeric_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [True, False],
    })
    # TODO(aksub99): Add tests for inf and NaN.
))
class TestMaximum(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    dodge_nondifferentiable = True

    def func(self, xp, a, b):
        return xp.maximum(a, b)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_arithmetic_invalid)
def test_maximum_invalid_dtypes(device, dtypes):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (3, 2)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        chainerx.maximum(a, b)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_binary,
        'in_dtypes,out_dtype': (
            math_utils.make_same_in_out_dtypes(
                2, chainerx.testing.numeric_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [False],
    })
    # is_module
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_dtypes_arithmetic,
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # TODO(aksub99): Add tests for inf and NaN.
))
class TestMinimum(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    dodge_nondifferentiable = True

    def func(self, xp, a, b):
        return xp.minimum(a, b)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_arithmetic_invalid)
def test_minimum_invalid_dtypes(device, dtypes):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (3, 2)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        chainerx.minimum(a, b)


_mean_var_params = \
    chainer.testing.product({
        'shape,axis': [
            ((), None),
            (1, 0),
            ((2, 1, 3), (1, 2)),
            ((1, 1, 1), (0, 1, 2)),
            ((2, 3), None),
            ((1, 2, 3), (0, 2)),
            ((2, 2, 2, 2), (2, 1, 0)),
            ((1, 1, 1), (-1))],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
        'input': ['random'],
        'contiguous': [None, 'C'],
    }) + chainer.testing.product({
        'shape,axis': [((2, 3), None)],
        'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
        'input': [1.57, 2, 3.14, float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _mean_var_params
))
class TestMean(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.mean(a, self.axis)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _mean_var_params
))
class TestVar(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.var(a, self.axis)


def apply_func(is_module, func, xp, device, input, axis, dtypes):
    (in_dtype,), out_dtype = dtypes
    try:
        a_np = input.astype(in_dtype)
    except (ValueError, OverflowError):
        return xp.zeros(())  # invalid combination of data and dtype

    a = xp.array(a_np)
    a = func(is_module, xp, a, axis)
    if xp is numpy:
        a = dtype_utils.cast_if_numpy_array(xp, a, out_dtype)
    return a


def compute_mean(is_module, xp, a, axis):
    return xp.mean(a, axis) if is_module else a.mean(axis)


def compute_var(is_module, xp, a, axis):
    return xp.var(a, axis) if is_module else a.var(axis)


@chainerx.testing.numpy_chainerx_array_equal(strides_check=False)
@pytest.mark.parametrize('input,axis', [
    # --- single axis
    # input, axis
    # valid params
    (numpy.asarray(0), None),
    (numpy.asarray(-1), None),
    (numpy.asarray(float('inf')), None),
    (numpy.asarray(float('nan')), None),
    (numpy.asarray(-float('inf')), None),
    (numpy.asarray([4, 1, 4, 1]), None),
    (numpy.asarray([4, 1, 4, 1]), 0),
    (numpy.asarray([[4, 4, 1, 1], [4, 1, 4, 1]]), 0),
    (numpy.asarray([[4, 4, 1, 1], [4, 1, 4, 1]]).T, 1),
    (numpy.asarray([-0.0, +0.0, +0.0, -0.0]), None),
    (numpy.asarray([[True, True, False, False],
                    [True, False, True, False]]), 0),
    (numpy.ones((2, 0, 3)), 2),
    (numpy.ones((2, 3)), 1),
    (numpy.ones((2, 3)), -2),
    # --- multiple axes
    # input, axis
    # valid params
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (0, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (-2, -1)),
])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_math_functions)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('func', [
    compute_mean,
    compute_var,
])
# TODO(kshitij12345): Remove strides_check=False
def test_valid_stats(is_module, func, xp, device, input, axis, dtypes):
    return apply_func(is_module, func, xp, device, input, axis, dtypes)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(IndexError, ValueError, chainerx.DimensionError),
    strides_check=False)
@pytest.mark.parametrize('input,axis', [
    # --- single axis
    # input, axis
    # invalid params
    (numpy.ones((0,)), None),
    (numpy.ones((2, 0, 3)), 1),
    (numpy.ones((2, 0, 3)), None),
    (numpy.ones((2, 3)), 2),
    (numpy.ones((2, 3)), -3),
    # --- multiple axes
    # input, axis
    # invalid params
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (-3, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 2)),
])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_math_functions)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('func', [
    compute_mean,
    compute_var,
])
# TODO(kshitij12345): Remove strides_check=False
def test_invalid_stats(is_module, func, xp, device, input, axis, dtypes):
    return apply_func(is_module, func, xp, device, input, axis, dtypes)
