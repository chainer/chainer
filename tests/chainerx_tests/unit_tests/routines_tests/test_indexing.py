import unittest

import numpy
import pytest

import chainer.testing
import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import math_utils
from chainerx_tests import op_utils


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shape,indices', [
    # empty indexing
    ((), ()),
    ((3,), ()),
    ((2, 2, 2), ()),
    # integer indexing - non-tuple indexing
    ((3,), 0),
    ((3,), 1),
    ((3,), 2),
    ((3,), -1),
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 3), numpy.int8(-1)),
    ((2, 3), numpy.int32(0)),
    ((2, 3), numpy.uint64(1)),
    # integer indexining - tuple indexing
    ((3,), (0,)),
    ((3,), (1,)),
    ((3,), (2,)),
    ((3,), (-1,)),
    ((2, 3), (0,)),
    ((2, 3), (1,)),
    ((2, 3), (0, 0)),
    ((2, 3), (1, 1)),
    ((2, 3, 4), (0, -2, 3)),
    ((2, 3, 4), (1, 0)),
    # slice indexing - non-tuple indexing
    ((3,), slice(None)),
    ((3,), slice(2)),
    ((3,), slice(0, 3)),
    ((3,), slice(0, 2)),
    ((3,), slice(1, 3)),
    ((3,), slice(0, 0)),
    ((3,), slice(0, 1)),
    ((3,), slice(2, 0, -1)),
    ((3,), slice(-2, -1)),
    ((3,), slice(2, None, -1)),
    ((3,), slice(None, 0, 1)),
    ((3,), slice(None, -1, -1)),
    ((3,), slice(None, -2, -1)),
    ((6,), slice(0, 6, 2)),
    ((6,), slice(1, 6, 2)),
    ((6,), slice(5, None, -2)),
    # slice indexing - tuple indexing
    ((3,), (slice(None),)),
    ((3,), (slice(2),)),
    ((3,), (slice(0, 3),)),
    ((3,), (slice(0, 2),)),
    ((3,), (slice(1, 3),)),
    ((3,), (slice(0, 0),)),
    ((3,), (slice(0, 1),)),
    ((3,), (slice(2, 0, -1),)),
    ((3,), (slice(-2, -1),)),
    ((3,), (slice(2, None, -1),)),
    ((3,), (slice(None, 0, 1),)),
    ((3,), (slice(None, -1, -1),)),
    ((3,), (slice(None, -2, -1),)),
    ((6,), (slice(0, 6, 2),)),
    ((6,), (slice(1, 6, 2),)),
    ((6,), (slice(5, None, -2),)),
    ((6,), (slice(50, 1, -1),)),
    ((6,), (slice(3, 3, 1),)),
    ((6,), (slice(3, 3, -2),)),
    ((6,), (slice(50, 50, 1),)),
    ((6,), (slice(50, 50, -2),)),
    ((6,), (slice(-50, -50, 1),)),
    ((6,), (slice(-50, -50, -2),)),
    ((2, 3), (slice(None), slice(None))),
    ((2, 3), (slice(1), slice(2))),
    ((2, 3), (slice(0, 2), slice(0, 3))),
    ((2, 3), (slice(0, 2), slice(0, -1))),
    ((2, 3), (slice(0, None, -1), slice(2, 3))),
    ((2, 3), (slice(0, None, None), slice(-2, 0, -1))),
    ((2, 3), (slice(1, 2), slice(0, 2))),
    ((2, 3), (slice(-2, None, -1), slice(0, 3))),
    ((2, 3), (slice(-2, None, -1), slice(-3, None, -1))),
    ((2, 3), (slice(-2, None, -1), slice(None, None, -2))),
    ((2, 3), (slice(1, 2), slice(None, None, 1))),
    ((2, 3), (slice(1, 2), slice(None, None, 2))),
    ((2, 3, 4), (slice(1), slice(-2, 3), slice(1, None, -1))),
    # newaxis indexing - non-tuple indexing
    ((), chainerx.newaxis),
    ((3,), chainerx.newaxis),
    # newaxis indexing - tuple indexing
    ((), (chainerx.newaxis,)),
    ((3,), (chainerx.newaxis,)),
    ((2, 3), (chainerx.newaxis, chainerx.newaxis)),
    # mixed indexing - tuple indexing
    ((2, 3), (0, slice(1, 3))),
    ((4, 3), (slice(1, 3), 1)),
    ((2, 3, 4), (1, slice(2,), slice(1, 3))),
    ((2, 3), (1, chainerx.newaxis, slice(1, 3))),
    ((2, 3, 4), (slice(0, 1), slice(1, 2), slice(1, 3), chainerx.newaxis)),
    ((2, 3, 4), (slice(0, 1), slice(1, 2), chainerx.newaxis, slice(1, 3))),
    ((2, 3, 4), (slice(0, 1), chainerx.newaxis, slice(1, 2), slice(1, 3))),
    ((2, 3, 4), (chainerx.newaxis, slice(0, 1), slice(1, 2), slice(1, 3))),
    ((2, 3, 4),
     (1, slice(2,), chainerx.newaxis, slice(1, 3), chainerx.newaxis)),
])
class TestGetitem(op_utils.NumpyOpTest):
    # TODO(niboshi): Remove this
    check_numpy_strides_compliance = False

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype('float32')
        return x,

    def forward_xp(self, inputs, xp):
        x, = inputs
        y = x[self.indices]
        return y,


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_getitem_zero_sized_offsets(device):
    a = chainerx.arange(6)

    b = a[3:3]
    # Test pre-conditions.
    assert b.size == 0
    assert b.offset == 12

    # The offset of `c` should be the same as `b` since `b` is empty.
    c = b[2:]
    assert c.size == 0
    assert c.offset == b.offset


@op_utils.op_test(['native:0', 'cuda:0'])
# TODO(hvy): Add cases where axis=None, when supported.
@chainer.testing.parameterize_pytest('shape,indices,axis', [
    # Valid parameters
    ((3,), [0], 0),
    ((3,), [1], 0),
    ((2, 3), [0], 0),
    ((2, 3), [0], 1),
    ((2, 3), [0], -1),
    ((2, 3), [1], 0),
    ((2, 3), [0, -1], 0),
    ((2, 3), [1, 0], 0),
    ((2, 3), [1, 2], 1),
    ((2, 3), [2, 1], 1),
    ((2, 3), [[0], [1]], 0),
    # Invalid: Axis out of bounds
    ((2, 3), [0], 2),
    ((2, 3), [0], -3),
])
@chainer.testing.parameterize_pytest('is_module', [True, False])
@chainer.testing.parameterize_pytest(
    'indices_type', ['list', 'numpy', 'xp'])
# TODO(niboshi): indices_dtype is ignored if indices_type == 'list', which is
# wasteful.
@chainer.testing.parameterize_pytest(
    'indices_dtype', chainerx.testing.integral_dtypes)
class TestTake(op_utils.NumpyOpTest):

    check_numpy_strides_compliance = False
    forward_accept_errors = (chainerx.DimensionError, numpy.AxisError)

    def setup(self):
        if (numpy.dtype(self.indices_dtype).kind == 'u'
                and (numpy.array(self.indices, 'int64') < 0).any()):
            raise unittest.SkipTest(
                'Indices underflows and index out of bounds cannot be tested.')

    def generate_inputs(self):
        a = numpy.random.uniform(-1, 1, self.shape).astype('float32')
        return a,

    def forward_xp(self, inputs, xp):
        indices = self.indices
        axis = self.axis
        indices_type = self.indices_type
        a, = inputs

        assert isinstance(indices, list)
        if indices_type == 'list':
            pass
        elif indices_type == 'numpy':
            indices = numpy.array(indices).astype(self.indices_dtype)
        elif indices_type == 'xp':
            indices = xp.array(indices).astype(self.indices_dtype)
        else:
            assert False, indices_type

        if self.is_module:
            b = xp.take(a, indices, axis)
        else:
            b = a.take(indices, axis)
        return b,


def _random_condition(shape, dtype):
    size = int(numpy.prod(shape))
    mask = numpy.random.randint(0, 1, size).astype('bool_').reshape(shape)
    pos = array_utils.uniform(shape, dtype)
    pos[numpy.logical_not(pos)] = True  # All elements are True
    return pos * mask


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'cond_shape,in_shapes': [
            # Same Shapes
            ((2, 3), ((2, 3), (2, 3))),
            # Broadcast Shapes
            ((2, 3), ((1, 3), (1, 3))),
            ((2, 3), ((2, 1), (1, 3))),
            ((2, 3), ((2, 3), (1, 3))),
            ((4, 5), ((3, 4, 1), (1, 5))),
            ((1, 4, 5), ((3, 4, 1), (3, 1, 5))),
        ],
        'cond_dtype': ['bool_'],
        'in_dtypes,out_dtype': dtype_utils.result_dtypes_two_arrays,
    })
    # Dtype combinations
    + chainer.testing.product({
        'cond_shape,in_shapes': [((2, 3), ((2, 3), (2, 3)))],
        'cond_dtype': chainerx.testing.all_dtypes,
        'in_dtypes,out_dtype': dtype_utils.result_dtypes_two_arrays,
    })
))
class TestWhere(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    check_numpy_strides_compliance = False
    dodge_nondifferentiable = True
    input_lhs = 'random'
    input_rhs = 'random'

    def generate_inputs(self):
        self.condition = _random_condition(self.cond_shape, self.cond_dtype)
        return super().generate_inputs()

    def func(self, xp, x, y):
        condition = xp.array(self.condition)
        return xp.where(condition, x, y)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(
        chainerx.DimensionError, ValueError))
@pytest.mark.parametrize('cond_shape,x_shape,y_shape', [
    ((2, 3), (3, 4), (2, 3)),
    ((2, 3), (2, 3), (3, 4)),
    ((2, 3), (1, 3), (2, 4))
])
def test_where_invalid_shapes(xp, cond_shape, x_shape, y_shape):
    x = array_utils.create_dummy_ndarray(xp, x_shape, 'float32')
    y = array_utils.create_dummy_ndarray(xp, y_shape, 'float32')
    c = array_utils.create_dummy_ndarray(xp, cond_shape, 'float32')
    return xp.where(c, x, y)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'cond_shape,shape': math_utils.shapes_combination_inplace_binary,
        'cond_dtype': ['bool_'],
        'in_dtypes,scalar_type,out_dtype': (
            dtype_utils.result_dtypes_array_scalar),
        'is_scalar_rhs': [True, False],
    })
    # Dtype combinations
    + chainer.testing.product({
        'cond_shape,shape': [((2, 3), (2, 3))],
        'cond_dtype': chainerx.testing.all_dtypes,
        'in_dtypes,scalar_type,out_dtype': (
            dtype_utils.result_dtypes_array_scalar),
        'is_scalar_rhs': [True, False],
    })
))
class TestWhereScalar(math_utils.MathScalarTestBase, op_utils.NumpyOpTest):

    check_numpy_strides_compliance = False
    input = 'random'
    scalar_value = 3

    def generate_inputs(self):
        self.condition = _random_condition(self.cond_shape, self.cond_dtype)
        return super().generate_inputs()

    def func_scalar(self, xp, a, scalar):
        condition = xp.array(self.condition)
        if self.is_scalar_rhs:
            return xp.where(condition, a, scalar)
        else:
            return xp.where(condition, scalar, a)


_in_out_dtypes_where_scalar = [
    ((bool, bool), 'bool_'),
    ((bool, int), 'int32'),
    ((bool, float), 'float32'),
    ((int, bool), 'int32'),
    ((int, int), 'int32'),
    ((int, float), 'float32'),
    ((float, bool), 'float32'),
    ((float, int), 'float32'),
    ((float, float), 'float32'),
]


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('cond_shape', [(2, 3)])
@pytest.mark.parametrize('cond_dtype', chainerx.testing.all_dtypes)
@pytest.mark.parametrize('in_types,out_dtype', _in_out_dtypes_where_scalar)
def test_where_scalar_scalar(xp, cond_shape, cond_dtype, in_types, out_dtype):
    cond = xp.array(_random_condition(cond_shape, cond_dtype))
    x_type, y_type = in_types
    x = x_type(0)
    y = y_type(2)
    out = xp.where(cond, x, y)
    return dtype_utils.cast_if_numpy_array(xp, out, out_dtype)
