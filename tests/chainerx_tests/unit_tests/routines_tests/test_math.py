import chainer
import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import math_utils
from chainerx_tests import op_utils


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
            dtype_utils.make_same_in_out_dtypes(
                1, chainerx.testing.numeric_dtypes)),
        'input': ['random'],
        'is_module': [False],
    })
    # is_module
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                1, chainerx.testing.numeric_dtypes)),
        'input': ['random'],
        'is_module': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.numeric_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.numeric_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.all_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
        'is_module': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
                1, chainerx.testing.float_dtypes)),
        'scalar_type': [float],
        'input': ['random'],
        'scalar_value': [1],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
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


def _create_dummy_array_for_dot(xp, shape, dtype):
    x = numpy.arange(numpy.prod(shape)).reshape(shape)
    if dtype == 'bool_':
        x = numpy.asarray(x % 2 == 0)
    else:
        x = x.astype(dtype)
    return xp.array(x)


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
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_binary,
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.numeric_dtypes)),
        'input_lhs,input_rhs': [(2, 2)],
        'is_module': [False],
    })
    # Dtype combinations
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': dtype_utils.result_numeric_dtypes_two_arrays,
        'input_lhs,input_rhs': [(2, 2)],
        'is_module': [False],
    })
    # is_module
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.numeric_dtypes)),
        'input_lhs,input_rhs': [(2, 2)],
        'is_module': [True, False],
    })
    # Special values (integers forward)
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.signed_integral_dtypes)),
        'input_lhs': [-2, -1, 0, 1, 2, 5],
        'input_rhs': [0, 1, 2, 5],
        'is_module': [False],
    })
    # Special values (floats forward)
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': [-1, 0, 1, 2, float('inf'), -float('inf'), float('nan')],
        'input_rhs': [-1, 0, 1, 2, float('inf'), -float('inf'), float('nan')],
        'is_module': [False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
    # Special values (floats backward)
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': [-3.0, -1.2, 1.2, 3],
        'input_rhs': [-3.0, -1.2, 0.0, 1.2, 3.0],
        'is_module': [False],
    })
))
class TestPower(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    def setup(self):
        super().setup()
        in_dtype1, in_dtype2 = self.in_dtypes
        if in_dtype1 == 'float16' or in_dtype2 == 'float16':
            self.check_backward_options.update({'rtol': 5e-3, 'atol': 5e-3})
            self.check_double_backward_options.update(
                {'rtol': 5e-3, 'atol': 5e-3})

    def func(self, xp, a, b):
        if self.is_module:
            y = xp.power(a, b)
        else:
            y = a ** b

        return y


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': [2],
        'scalar_value': [1.2, 2],
        'is_module': [False],
        'is_scalar_rhs': [True, False],
    })
    # Type combinations
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': [2],
        'scalar_value': [1.2, 2],
        'is_module': [False],
        'is_scalar_rhs': [True, False],
    })
    # is_module
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_arithmetic_scalar,
        'input': [2],
        'scalar_value': [1.2, 2],
        'is_module': [True, False],
        'is_scalar_rhs': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_float_arithmetic_scalar,
        'input': [-1, 0, 1, 2, float('inf'), -float('inf'), float('nan')],
        'scalar_value': [
            -1, 0, 1, 2, float('inf'), -float('inf'), float('nan')],
        'is_module': [False],
        'is_scalar_rhs': [False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestPowerScalar(math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    def setup(self):
        super().setup()
        if self.in_dtypes == 'float16':
            self.check_backward_options.update({'rtol': 5e-3, 'atol': 5e-3})
            self.check_double_backward_options.update(
                {'rtol': 5e-3, 'atol': 5e-3})

    def func_scalar(self, xp, a, scalar):
        if self.is_module:
            if self.is_scalar_rhs:
                y = xp.power(a, scalar)
            else:
                y = xp.power(scalar, a)
        else:
            if self.is_scalar_rhs:
                y = a ** scalar
            else:
                y = scalar ** a

        return y


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtype', chainerx.testing.all_dtypes)
@pytest.mark.parametrize('is_bool_rhs', [True, False])
@pytest.mark.parametrize('is_bool_primitive', [True, False])
@pytest.mark.parametrize('is_module', [True, False])
def test_power_invalid_bool_dtype(
        device, dtype, is_bool_rhs, is_bool_primitive, is_module):
    shape = (3, 2)

    a = chainerx.array(array_utils.uniform(shape, dtype))

    if is_bool_primitive:
        b = True
    else:
        b = chainerx.array(array_utils.uniform(shape, 'bool'))

    with pytest.raises(chainerx.DtypeError):
        if is_module:
            if is_bool_rhs:
                chainerx.power(a, b)
            else:
                chainerx.power(b, a)
        else:
            if is_bool_rhs:
                a ** b
            else:
                b ** a


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
@pytest.mark.parametrize('dtypes', math_utils.in_out_dtypes_math_functions)
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
