import chainer
import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import math_utils
from chainerx_tests import op_utils


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

_in_out_dtypes_inplace_bitwise_invalid = [
    (('bool_', 'int8'), 'int8'),
    (('bool_', 'int16'), 'int16'),
    (('bool_', 'int32'), 'int32'),
    (('bool_', 'int64'), 'int64'),
    (('bool_', 'uint8'), 'uint8'),
]

_in_out_dtypes_inplace_bitwise = [
    dtypes for dtypes in _in_out_dtypes_bitwise
    if dtypes not in _in_out_dtypes_inplace_bitwise_invalid
]

_in_out_dtypes_bitwise_scalar = [
    # Bool scalar
    # TODO(imanishi): Support bool in op_utils.NumpyOpTest
    # (('bool_',), bool, 'bool_'),
    (('int8',), bool, 'int8'),
    (('int32',), bool, 'int32'),
    (('uint8',), bool, 'uint8'),
    (('uint8',), numpy.bool_, 'uint8'),
    # Int scalar
    # TODO(imanishi): Support bool in op_utils.NumpyOpTest
    # (('bool_',), int, 'bool_'),
    (('int8',), int, 'int8'),
    (('int16',), int, 'int16'),
    (('int32',), int, 'int32'),
    (('int64',), int, 'int64'),
    (('uint8',), int, 'uint8'),
    (('int16',), numpy.int16, 'int16'),
    (('uint8',), numpy.int8, 'uint8'),
]

_scalar_invalid_bitwise = [
    (('float16',), int, 'float16'),
    (('float32',), int, 'float32'),
    (('float64',), int, 'float64'),
    (('float64',), numpy.int8, 'float64'),
    (('float16',), numpy.int64, 'float16'),
]

_in_out_dtypes_shift = [
    (('int8', 'int8'), 'int8'),
    (('int16', 'int16'), 'int16'),
    (('int32', 'int32'), 'int32'),
    (('int64', 'int64'), 'int64'),
    (('uint8', 'uint8'), 'uint8'),
    (('int8', 'int16'), 'int8'),
    (('int8', 'int32'), 'int8'),
    (('int8', 'int64'), 'int8'),
    (('int8', 'uint8'), 'int8'),
    (('int16', 'int32'), 'int16'),
    (('int16', 'int64'), 'int16'),
    (('int16', 'uint8'), 'int16'),
    (('int32', 'int64'), 'int32'),
    (('int32', 'uint8'), 'int32'),
    (('int64', 'uint8'), 'int64'),
]

_in_out_dtypes_shift_array_scalar = [
    (('int8',), int, 'int8'),
    (('int16',), int, 'int16'),
    (('int32',), int, 'int32'),
    (('int64',), int, 'int64'),
    (('uint8',), int, 'uint8'),
    (('int16',), numpy.int16, 'int16'),
    (('uint8',), numpy.int8, 'uint8'),
]

_in_out_dtypes_shift_scalar_array = [
    (('int8',), int, 'int64'),
    (('int16',), int, 'int64'),
    (('int32',), int, 'int64'),
    (('int64',), int, 'int64'),
    (('uint8',), int, 'int64'),
    (('int16',), numpy.int16, 'int64'),
    (('uint8',), numpy.int8, 'int64'),
]

_in_out_dtypes_shift_invalid = [
    (('bool_', 'int8'), 'int8'),
    (('bool_', 'int16'), 'int16'),
    (('bool_', 'int32'), 'int32'),
    (('bool_', 'int64'), 'int64'),
    (('bool_', 'uint8'), 'uint8'),
    (('float', 'int8'), 'int8'),
    (('float', 'int16'), 'int16'),
    (('float', 'int32'), 'int32'),
    (('float', 'int64'), 'int64'),
    (('float', 'uint8'), 'uint8'),
]

_params_bitwise = (
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_binary,
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
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
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.nonfloat_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # Dtype combinations
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_dtypes_inplace_bitwise,
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # Special values
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.nonfloat_dtypes)),
        'input_lhs': ['random', float('inf'), -float('inf'), float('nan')],
        'input_rhs': ['random', float('inf'), -float('inf'), float('nan')],
    })
)


_scalar_params_bitwise = (
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_bitwise_scalar,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [False],
    })
    # Type combinations
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_bitwise_scalar,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [True, False],
    })
    # is_module
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_bitwise_scalar,
        'input': ['random'],
        'scalar_value': [1],
        'is_module': [True, False],
        'is_scalar_rhs': [True, False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_bitwise_scalar,
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
            _in_out_dtypes_bitwise_scalar,
        'input': ['random'],
        'scalar_value': [1],
    })
    # Dtype combinations
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_bitwise_scalar,
        'input': ['random'],
        'scalar_value': [1],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_bitwise_scalar,
        'input': [float('inf'), -float('inf'), float('nan')],
        'scalar_value': [
            0, -1, 1, 2],
    })
)

_params_shift = (
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_binary,
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.integral_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': [0, 1, 3],
        'is_module': [False],
    })
    # Dtype combinations
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_dtypes_shift,
        'input_lhs': ['random'],
        'input_rhs': [0, 1, 3],
        'is_module': [False],
    })
    # is_module
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.integral_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': [0, 1, 3],
        'is_module': [True, False],
    })
)

_inplace_params_shift = (
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_inplace_binary,
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.integral_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': [0, 1, 3],
    })
    # Dtype combinations
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': _in_out_dtypes_shift,
        'input_lhs': ['random'],
        'input_rhs': [0, 1, 3],
    })
)

_scalar_params_shift_scalar_array = (
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_shift_scalar_array,
        'input': [0, 1, 3],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [False],
    })
    # Type combinations
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_shift_scalar_array,
        'input': [0, 1, 3],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [False],
    })
    # is_module
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_shift_scalar_array,
        'input': [0, 1, 3],
        'scalar_value': [1],
        'is_module': [True, False],
        'is_scalar_rhs': [False],
    })
)

_scalar_params_shift_array_scalar = (
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_shift_array_scalar,
        'input': [0, 1, 3],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [True],
    })
    # Type combinations
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_shift_array_scalar,
        'input': [0, 1, 3],
        'scalar_value': [1],
        'is_module': [False],
        'is_scalar_rhs': [True],
    })
    # is_module
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype': _in_out_dtypes_shift_array_scalar,
        'input': [0, 1, 3],
        'scalar_value': [1],
        'is_module': [True, False],
        'is_scalar_rhs': [True],
    })
)

_inplace_scalar_params_shift = (
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_shift_array_scalar,
        'input': [0, 1, 3],
        'scalar_value': [1],
    })
    # Dtype combinations
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,scalar_type,out_dtype':
            _in_out_dtypes_shift_array_scalar,
        'input': [0, 1, 3],
        'scalar_value': [1],
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
@chainer.testing.parameterize(*_params_shift)
class TestLeftShift(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        if self.is_module:
            return xp.left_shift(a, b)
        else:
            return a << b


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_params_shift)
class TestRightShift(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        if self.is_module:
            return xp.right_shift(a, b)
        else:
            return a >> b


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


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_inplace_params_shift)
class TestILeftShift(
        math_utils.InplaceBinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        a <<= b


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_inplace_params_shift)
class TestIRightShift(
        math_utils.InplaceBinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        a >>= b


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_inplace_bitwise_invalid)
def test_iand_invalid_dtypes(device, dtypes):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (2, 3)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        a &= b


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_inplace_bitwise_invalid)
def test_ior_invalid_dtypes(device, dtypes):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (2, 3)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        a |= b


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_inplace_bitwise_invalid)
def test_ixor_invalid_dtypes(device, dtypes):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (2, 3)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        a ^= b


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_shift_invalid)
def test_ileftshift_invalid_dtypes(device, dtypes):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (2, 3)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        a << b


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtypes', _in_out_dtypes_shift_invalid)
def test_irightshift_invalid_dtypes(device, dtypes):
    (in_dtype1, in_dtype2), _ = dtypes
    shape = (2, 3)
    a = chainerx.array(array_utils.uniform(shape, in_dtype1))
    b = chainerx.array(array_utils.uniform(shape, in_dtype2))
    with pytest.raises(chainerx.DtypeError):
        a >> b


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
class TestBitwiseOrScalar(
        math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

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
@chainer.testing.parameterize(*_scalar_params_shift_scalar_array)
class TestLeftShiftScalarArray(
        math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        if self.is_module:
            return scalar << a
        else:
            return xp.left_shift(scalar, a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_scalar_params_shift_array_scalar)
class TestLeftShiftArrayScalar(
        math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        if self.is_module:
            return a << scalar
        else:
            return xp.left_shift(a, scalar)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_scalar_params_shift_scalar_array)
class TestRightShiftScalarArray(
        math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        if self.is_module:
            return scalar >> a
        else:
            return xp.right_shift(scalar, a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_scalar_params_shift_array_scalar)
class TestRightShiftArrayScalar(
        math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        if self.is_module:
            return a >> scalar
        else:
            return xp.right_shift(a, scalar)


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
@chainer.testing.parameterize(*_scalar_params_shift_array_scalar)
class TestILeftShiftScalar(
        math_utils.InplaceMathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        a <<= scalar


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*_scalar_params_shift_array_scalar)
class TestIRightShiftScalar(
        math_utils.InplaceMathScalarTestBase, op_utils.NumpyOpTest):

    def func_scalar(self, xp, a, scalar):
        a >>= scalar
