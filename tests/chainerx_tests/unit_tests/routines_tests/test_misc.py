import chainer
import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import math_utils
from chainerx_tests import op_utils


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (1,), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
        'input': [1, 3],
    })
    # Special shapes (array.size = 0)
    + chainer.testing.product({
        'shape': [(0,), (2, 0, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
        'input': [1, 3],
        'check_numpy_strides_compliance': [False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
        'input': [float('inf'), -float('inf'), float('nan'), -1, 0],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestSqrt(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.sqrt(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': dtype_utils.make_same_in_out_dtypes(
            1, chainerx.testing.numeric_dtypes),
        'input': ['random'],
        'contiguous': [None, 'C'],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': dtype_utils.make_same_in_out_dtypes(
            1, chainerx.testing.float_dtypes),
        'input': [float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestSquare(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.square(a)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_square_invalid_dtypes(device):
    shape = (3, 2)
    bool_array = chainerx.array(array_utils.uniform(shape, 'bool_'))
    with pytest.raises(chainerx.DtypeError):
        chainerx.square(bool_array)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': ['random'],
        'input_rhs': ['random'],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
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
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
        'input': ['random'],
        'contiguous': [None, 'C'],
        'is_module': [True, False],
    })
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
        'input': [float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
        'is_module': [True, False],
    })
))
class TestAbs(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    dodge_nondifferentiable = True

    def func(self, xp, a):
        # Check correct alias.
        assert chainerx.abs is chainerx.absolute

        # Check computed result.
        if self.is_module:
            return xp.abs(a)
        else:
            return abs(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
        'input': [-2.5, -1.5, -0.1, 0.1, 1.5, 2.5],
        'contiguous': [None, 'C'],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
        'input': [float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestFabs(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.fabs(a)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0.5),
    numpy.asarray(-1.2),
    numpy.asarray(10.9),
    numpy.asarray(-10.6),
    numpy.asarray(0.),
    numpy.asarray(float('inf')),
    numpy.asarray(-float('inf')),
    numpy.asarray(float('nan')),
    numpy.full((), 2.1),
    numpy.full((0,), 2),
    numpy.full((2, 3), 0),
    numpy.full((2, 3), 2.6),
    numpy.full((1, 1), -1.01),
    numpy.full((1, 1), 1.99),
])
@pytest.mark.parametrize('dtypes', [
    (('int8',), 'int8'),
    (('int16',), 'int16'),
    (('int32',), 'int32'),
    (('int64',), 'int64'),
    (('float16',), 'float16'),
    (('float32',), 'float32'),
    (('float64',), 'float64'),
])
def test_sign(xp, device, input, dtypes):
    (in_dtype, ), out_dtype = dtypes
    a = xp.array(input.astype(in_dtype))
    return xp.sign(a)


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
        'in_dtypes,out_dtype': dtype_utils.result_comparable_dtypes_two_arrays,
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
    # TODO(aksub99): Add tests for inf and NaN.
))
class TestMaximum(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    dodge_nondifferentiable = True

    def func(self, xp, a, b):
        return xp.maximum(a, b)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtype', chainerx.testing.numeric_dtypes)
def test_maximum_invalid_dtypes(device, dtype):
    shape = (3, 2)
    bool_array = chainerx.array(array_utils.uniform(shape, 'bool_'))
    numeric_array = chainerx.array(array_utils.uniform(shape, dtype))
    with pytest.raises(chainerx.DtypeError):
        chainerx.maximum(bool_array, numeric_array)
    with pytest.raises(chainerx.DtypeError):
        chainerx.maximum(numeric_array, bool_array)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype': (
            dtype_utils.result_comparable_dtypes_array_scalar),
        'input': ['random'],
        'scalar_value': [0, 1],
        'is_scalar_rhs': [False],
    })
    # Differentiable cases
    + chainer.testing.product({
        'in_dtypes,scalar_type,out_dtype': (
            dtype_utils.result_comparable_dtypes_array_scalar),
        'input': [numpy.array([1, 3, 3, 4])],
        'scalar_value': [0, 2, 5],
        'is_scalar_rhs': [False, True],
    })
    # Non-differentiable cases
    + chainer.testing.product({
        'in_dtypes,scalar_type,out_dtype': (
            dtype_utils.result_comparable_dtypes_array_scalar),
        'input': [numpy.array([1, 3, 3, 4])],
        'scalar_value': [1, 3, 4],
        'is_scalar_rhs': [False, True],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
    # Special float values
    + chainer.testing.product({
        'in_dtypes,scalar_type,out_dtype': (
            dtype_utils.result_float_dtypes_array_scalar),
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
        'in_dtypes,out_dtype': dtype_utils.result_comparable_dtypes_two_arrays,
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
    # TODO(aksub99): Add tests for inf and NaN.
))
class TestMinimum(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    dodge_nondifferentiable = True

    def func(self, xp, a, b):
        return xp.minimum(a, b)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtype', chainerx.testing.numeric_dtypes)
def test_minimum_invalid_dtypes(device, dtype):
    shape = (3, 2)
    bool_array = chainerx.array(array_utils.uniform(shape, 'bool_'))
    numeric_array = chainerx.array(array_utils.uniform(shape, dtype))
    with pytest.raises(chainerx.DtypeError):
        chainerx.minimum(bool_array, numeric_array)
    with pytest.raises(chainerx.DtypeError):
        chainerx.minimum(numeric_array, bool_array)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,scalar_type,out_dtype': (
            dtype_utils.result_comparable_dtypes_array_scalar),
        'input': ['random'],
        'scalar_value': [1],
        'is_scalar_rhs': [False],
    })
    # Differentiable cases
    + chainer.testing.product({
        'in_dtypes,scalar_type,out_dtype': (
            dtype_utils.result_comparable_dtypes_array_scalar),
        'input': [numpy.array([1, 3, 3, 4])],
        'scalar_value': [0, 2, 5],
        'is_scalar_rhs': [False, True],
    })
    # Non-differentiable cases
    + chainer.testing.product({
        'in_dtypes,scalar_type,out_dtype': (
            dtype_utils.result_comparable_dtypes_array_scalar),
        'input': [numpy.array([1, 3, 3, 4])],
        'scalar_value': [1, 3, 4],
        'is_scalar_rhs': [False, True],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
    # Special float values
    + chainer.testing.product({
        'in_dtypes,scalar_type,out_dtype': (
            dtype_utils.result_float_dtypes_array_scalar),
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
