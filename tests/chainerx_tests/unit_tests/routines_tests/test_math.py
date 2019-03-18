import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import op_utils


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_negative(xp, device, shape, dtype, is_module):
    if dtype == 'bool_':  # Checked in test_invalid_bool_neg
        return chainerx.testing.ignore()
    x = array_utils.create_dummy_ndarray(xp, shape, dtype)
    if is_module:
        return xp.negative(x)
    else:
        return -x


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(chainerx.DtypeError, TypeError))
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_negative_invalid_bool(xp, device, is_module):
    x = xp.array([True, False], dtype='bool_')
    if is_module:
        xp.negative(x)
    else:
        -x


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_add(xp, device, shape, dtype, is_module):
    lhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=2)
    if is_module:
        return xp.add(lhs, rhs)
    else:
        return lhs + rhs


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_iadd(xp, device, shape, dtype):
    lhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=2)
    lhs += rhs
    return lhs


@pytest.mark.parametrize('scalar', [0, -1, 1, 2, 0.0, -1.0, 1.0, 2.0])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_add_scalar(scalar, device, shape, dtype):
    x_np = array_utils.create_dummy_ndarray(numpy, shape, dtype)
    # Implicit casting in NumPy's multiply depends on the 'casting' argument,
    # which is not yet supported (ChainerX always casts).
    # Therefore, we explicitly cast the scalar to the dtype of the ndarray
    # before the multiplication for NumPy.
    expected = x_np + numpy.dtype(dtype).type(scalar)

    x = chainerx.array(x_np)
    scalar_chx = chainerx.Scalar(scalar)
    chainerx.testing.assert_array_equal_ex(x + scalar, expected)
    chainerx.testing.assert_array_equal_ex(x + scalar_chx, expected)
    chainerx.testing.assert_array_equal_ex(scalar + x, expected)
    chainerx.testing.assert_array_equal_ex(scalar_chx + x, expected)
    chainerx.testing.assert_array_equal_ex(chainerx.add(x, scalar), expected)
    chainerx.testing.assert_array_equal_ex(
        chainerx.add(x, scalar_chx), expected)
    chainerx.testing.assert_array_equal_ex(chainerx.add(scalar, x), expected)
    chainerx.testing.assert_array_equal_ex(
        chainerx.add(scalar_chx, x), expected)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('scalar', [0, -1, 1, 2])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_iadd_scalar(xp, scalar, device, shape, dtype):
    lhs = array_utils.create_dummy_ndarray(xp, shape, dtype)
    rhs = scalar
    if xp is numpy:
        rhs = numpy.dtype(dtype).type(rhs)
    lhs += rhs
    return lhs


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_sub(xp, device, shape, numeric_dtype, is_module):
    lhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype, pattern=2)
    if is_module:
        return xp.subtract(lhs, rhs)
    else:
        return lhs - rhs


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_isub(xp, device, shape, numeric_dtype):
    lhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype, pattern=2)
    lhs -= rhs
    return lhs


@pytest.mark.parametrize('scalar', [0, -1, 1, 2, 0.0, -1.0, 1.0, 2.0])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_sub_scalar(scalar, device, shape, dtype):
    if dtype == 'bool_':
        # Boolean subtract is deprecated.
        return chainerx.testing.ignore()
    x_np = array_utils.create_dummy_ndarray(numpy, shape, dtype)
    # Implicit casting in NumPy's multiply depends on the 'casting' argument,
    # which is not yet supported (ChainerX always casts).
    # Therefore, we explicitly cast the scalar to the dtype of the ndarray
    # before the multiplication for NumPy.
    expected = x_np - numpy.dtype(dtype).type(scalar)
    expected_rev = numpy.dtype(dtype).type(scalar) - x_np

    x = chainerx.array(x_np)
    scalar_chx = chainerx.Scalar(scalar)
    chainerx.testing.assert_array_equal_ex(x - scalar, expected)
    chainerx.testing.assert_array_equal_ex(x - scalar_chx, expected)
    chainerx.testing.assert_array_equal_ex(scalar - x, expected_rev)
    chainerx.testing.assert_array_equal_ex(scalar_chx - x, expected_rev)
    chainerx.testing.assert_array_equal_ex(
        chainerx.subtract(x, scalar), expected)
    chainerx.testing.assert_array_equal_ex(
        chainerx.subtract(x, scalar_chx), expected)
    chainerx.testing.assert_array_equal_ex(
        chainerx.subtract(scalar, x), expected_rev)
    chainerx.testing.assert_array_equal_ex(
        chainerx.subtract(scalar_chx, x), expected_rev)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('scalar', [0, -1, 1, 2])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_isub_scalar(xp, scalar, device, shape, dtype):
    if dtype == 'bool_':
        # Boolean subtract is deprecated.
        return chainerx.testing.ignore()
    lhs = array_utils.create_dummy_ndarray(xp, shape, dtype)
    rhs = scalar
    if xp is numpy:
        rhs = numpy.dtype(dtype).type(rhs)
    lhs -= rhs
    return lhs


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_mul(xp, device, shape, dtype, is_module):
    lhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=2)
    if is_module:
        return xp.multiply(lhs, rhs)
    else:
        return lhs * rhs


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_imul(xp, device, shape, dtype):
    lhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=2)
    lhs *= rhs
    return lhs


@pytest.mark.parametrize('scalar', [0, -1, 1, 2, 0.0, -1.0, 1.0, 2.0])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_mul_scalar(scalar, device, shape, dtype):
    x_np = array_utils.create_dummy_ndarray(numpy, shape, dtype)
    # Implicit casting in NumPy's multiply depends on the 'casting' argument,
    # which is not yet supported (ChainerX always casts).
    # Therefore, we explicitly cast the scalar to the dtype of the ndarray
    # before the multiplication for NumPy.
    expected = x_np * numpy.dtype(dtype).type(scalar)

    x = chainerx.array(x_np)
    scalar_chx = chainerx.Scalar(scalar)
    chainerx.testing.assert_array_equal_ex(x * scalar, expected)
    chainerx.testing.assert_array_equal_ex(x * scalar_chx, expected)
    chainerx.testing.assert_array_equal_ex(scalar * x, expected)
    chainerx.testing.assert_array_equal_ex(scalar_chx * x, expected)
    chainerx.testing.assert_array_equal_ex(
        chainerx.multiply(x, scalar), expected)
    chainerx.testing.assert_array_equal_ex(
        chainerx.multiply(x, scalar_chx), expected)
    chainerx.testing.assert_array_equal_ex(
        chainerx.multiply(scalar, x), expected)
    chainerx.testing.assert_array_equal_ex(
        chainerx.multiply(scalar_chx, x), expected)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('scalar', [0, -1, 1, 2])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_imul_scalar(xp, scalar, device, shape, dtype):
    lhs = array_utils.create_dummy_ndarray(xp, shape, dtype)
    rhs = scalar
    if xp is numpy:
        rhs = numpy.dtype(dtype).type(rhs)
    lhs *= rhs
    return lhs


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
def test_floordiv(xp, lhs, rhs, device, numeric_dtype, is_module):
    if (numpy.array(lhs).dtype.kind == 'f' and
            numpy.dtype(numeric_dtype).kind in ('i', 'u')):
        return chainerx.testing.ignore()
    if (((numpy.array(lhs) < 0).any() or (numpy.array(rhs) < 0).any()) and
            numpy.dtype(numeric_dtype).kind == 'u'):
        return chainerx.testing.ignore()
    lhs = xp.array(lhs).astype(numeric_dtype)
    if isinstance(rhs, (list, tuple)):
        rhs = xp.array(rhs).astype(numeric_dtype)

    if is_module:
        return xp.floor_divide(lhs, rhs)
    else:
        return lhs // rhs


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


@chainerx.testing.numpy_chainerx_array_equal(strides_check=False)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_truediv(xp, device, shape, numeric_dtype, is_module):
    lhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype)
    rhs = xp.arange(1, lhs.size + 1, dtype=numeric_dtype).reshape(shape)
    # TODO(beam2d): Remove astype after supporting correct dtype promotion.
    if is_module:
        return xp.divide(lhs, rhs)
    else:
        return lhs / rhs


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_itruediv(xp, device, shape, float_dtype):
    # TODO(niboshi): Remove padding=False
    lhs = array_utils.create_dummy_ndarray(
        xp, shape, float_dtype, padding=False)
    rhs = xp.arange(1, lhs.size + 1, dtype=float_dtype).reshape(shape)
    lhs /= rhs
    return lhs


# TODO(hvy): Support and test zero division and mixed dtypes (dtype kinds).
# TODO(hvy): Support and test chainerx.Scalar / chainerx.ndarray.
@pytest.mark.parametrize('scalar', [1, 2, -1.0, -2.0])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_truediv_scalar(scalar, device, shape, numeric_dtype):
    x_np = array_utils.create_dummy_ndarray(numpy, shape, numeric_dtype)
    expected = x_np / scalar

    x = chainerx.array(x_np)
    scalar_chx = chainerx.Scalar(scalar)
    chainerx.testing.assert_array_equal_ex(
        x / scalar, expected, strides_check=False)
    chainerx.testing.assert_array_equal_ex(
        x / scalar_chx, expected, strides_check=False)
    chainerx.testing.assert_array_equal_ex(
        chainerx.divide(x, scalar), expected, strides_check=False)
    chainerx.testing.assert_array_equal_ex(
        chainerx.divide(x, scalar_chx), expected, strides_check=False)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('scalar', [1, 2])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_itruediv_scalar(xp, scalar, device, shape, float_dtype):
    # TODO(niboshi): Remove padding=False
    lhs = array_utils.create_dummy_ndarray(
        xp, shape, float_dtype, padding=False)
    rhs = scalar
    lhs /= rhs
    return lhs


# TODO(niboshi): Remove strides_check=False
@chainerx.testing.numpy_chainerx_array_equal(strides_check=False)
@pytest.mark.parametrize('keepdims', [False, True])
@pytest.mark.parametrize('shape,axis', [
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
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_sum(is_module, xp, device, shape, axis, keepdims, dtype):
    a = array_utils.create_dummy_ndarray(xp, shape, dtype)
    if is_module:
        out = xp.sum(a, axis=axis, keepdims=keepdims)
    else:
        out = a.sum(axis=axis, keepdims=keepdims)

    # TODO(niboshi): Unsigned integer dtypes should result in uint64.
    # Currently chainerx returns int64.
    if xp is numpy and numpy.dtype(dtype).kind == 'u':
        out = out.astype(numpy.int64)
    return out


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


# TODO(sonots): Fix type compatibility for when shape is ()
@chainerx.testing.numpy_chainerx_array_equal(dtype_check=False)
@pytest.mark.parametrize('shape,value', [
    ((), -1),
    ((), 1),
    ((1,), -1),
    ((1,), 1),
    ((2,), 1),
    ((2, 3), 3),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_maximum_with_scalar(xp, device, shape, value, signed_dtype):
    a = array_utils.create_dummy_ndarray(xp, shape, signed_dtype)
    return xp.maximum(a, value)


def _create_dummy_array_for_dot(xp, shape, dtype):
    x = numpy.arange(numpy.prod(shape)).reshape(shape)
    if dtype == 'bool_':
        x = numpy.asarray(x % 2 == 0)
    else:
        x = x.astype(dtype)
    return xp.array(x)


# An association list that associates a dtype to the type which ChainerX's
# real-valued functions should return.
_expected_dtypes_math_functions = [
    # Float.
    ('float16', 'float16'),
    ('float32', 'float32'),
    ('float64', 'float64'),
    # Signed int.
    ('int8', 'float32'),
    ('int16', 'float32'),
    ('int32', 'float32'),
    ('int64', 'float32'),
    # Unsigned int.
    ('uint8', 'float32'),
    # Bool.
    ('bool_', 'float32'),
]


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0), numpy.asarray(-4), numpy.asarray(4),
    numpy.asarray(-float('inf')), numpy.asarray(float('inf')),
    numpy.asarray(float('nan')),
    numpy.full((), 2), numpy.full((0,), 2), numpy.full((2, 3), 2)
])
@pytest.mark.parametrize('in_dtype,out_dtype', _expected_dtypes_math_functions)
def test_exp(xp, device, input, in_dtype, out_dtype):
    a = xp.array(input.astype(in_dtype))
    a = dtype_utils.cast_if_numpy_array(xp, a, out_dtype)
    return xp.exp(a)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0), numpy.asarray(-1), numpy.asarray(1), numpy.asarray(
        10), numpy.asarray(float('inf')), numpy.asarray(float('nan')),
    numpy.full((), 2), numpy.full((0,), 2), numpy.full((2, 3), 2)
])
@pytest.mark.parametrize('in_dtype,out_dtype', _expected_dtypes_math_functions)
def test_log(xp, device, input, in_dtype, out_dtype):
    a = xp.array(input.astype(in_dtype))
    a = dtype_utils.cast_if_numpy_array(xp, a, out_dtype)
    return xp.log(a)


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


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _logsumexp_params)
@pytest.mark.parametrize('keepdims', [True, False])
@chainerx.testing.numpy_chainerx_allclose(float16_rtol=1e-3)
@pytest.mark.parametrize('in_dtype,out_dtype', _expected_dtypes_math_functions)
def test_logsumexp(xp, device, a_shape, axis, keepdims, in_dtype, out_dtype):
    a = array_utils.create_dummy_ndarray(xp, a_shape, in_dtype)
    if xp is numpy:
        a = a.astype(out_dtype)
        return numpy.log(numpy.exp(a).sum(axis=axis, keepdims=keepdims))
    return chainerx.logsumexp(a, axis=axis, keepdims=keepdims)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _invalid_logsumexp_params)
@pytest.mark.parametrize('keepdims', [True, False])
# TODO(hvy): Should not overflow for large numbers, add tests
def test_logsumexp_invalid(device, a_shape, axis, keepdims, dtype):
    a = array_utils.create_dummy_ndarray(chainerx, a_shape, dtype)
    with pytest.raises(chainerx.DimensionError):
        chainerx.logsumexp(a, axis=axis, keepdims=keepdims)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _logsumexp_params)
@chainerx.testing.numpy_chainerx_allclose(
    atol=1e-5, float16_rtol=3e-3, dtype_check=False)
@pytest.mark.parametrize('in_dtype,out_dtype', _expected_dtypes_math_functions)
def test_log_softmax(xp, device, a_shape, axis, in_dtype, out_dtype):
    a = array_utils.create_dummy_ndarray(xp, a_shape, in_dtype)
    if xp is numpy:
        a = a.astype(out_dtype)
        # Default is the second axis
        axis = axis if axis is not None else 1
        return a - xp.log(xp.sum(xp.exp(a), axis=axis, keepdims=True))
    return xp.log_softmax(a, axis=axis)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _invalid_logsumexp_params)
def test_log_softmax_invalid(device, a_shape, axis, dtype):
    a = array_utils.create_dummy_ndarray(chainerx, a_shape, dtype)
    with pytest.raises(chainerx.DimensionError):
        return chainerx.log_softmax(a, axis=axis)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0), numpy.asarray(-4), numpy.asarray(4),
    numpy.asarray(-float('inf')), numpy.asarray(float('inf')
                                                ), numpy.asarray(float('nan')),
    numpy.full((), 2), numpy.full((0,), 2), numpy.full((2, 3), 2)
])
@pytest.mark.parametrize('in_dtype,out_dtype', _expected_dtypes_math_functions)
def test_sqrt(xp, device, input, in_dtype, out_dtype):
    if (input.size > 0 and not numpy.isfinite(input).all() and
            numpy.dtype(in_dtype).kind != 'f'):
        return chainerx.testing.ignore()

    a = xp.array(input.astype(in_dtype))
    a = dtype_utils.cast_if_numpy_array(xp, a, out_dtype)
    return xp.sqrt(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0), numpy.asarray(-1), numpy.asarray(1), numpy.asarray(
        10), numpy.asarray(float('inf')), numpy.asarray(float('nan')),
    numpy.full((), 2), numpy.full((0,), 2), numpy.full((2, 3), 2)
])
@pytest.mark.parametrize('contiguous', [None, 'C'])
@pytest.mark.parametrize('in_dtype,out_dtype', _expected_dtypes_math_functions)
class TestTanh(op_utils.NumpyOpTest):

    def setup(self, input, contiguous, in_dtype, out_dtype):
        self.input = input.astype(in_dtype)
        self.chx_dtype = out_dtype
        self.contiguous = contiguous

        if in_dtype == 'float16':
            self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-2}
        if numpy.dtype(in_dtype).kind != 'f':
            self.skip_backward_test = True
            self.skip_double_backward_test = True

    def generate_inputs(self):
        return self.input,

    def forward_xp(self, inputs, xp):
        x, = inputs
        x = dtype_utils.cast_if_numpy_array(xp, x, self.chx_dtype)
        return xp.tanh(x),


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


def test_max_amax():
    assert chainerx.amax is chainerx.max


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(ValueError, chainerx.DimensionError), strides_check=False)
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
    # invalid params
    (numpy.ones((0,)), None),
    (numpy.ones((2, 0, 3)), 1),
    (numpy.ones((2, 0, 3)), None),
    (numpy.ones((2, 3)), 2),
    (numpy.ones((2, 3)), -3),
    # --- multiple axes
    # input, axis
    # valid params
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (0, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (-2, -1)),
    # invalid params
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (-3, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 2)),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
# TODO(niboshi): Remove strides_check=False
def test_max(is_module, xp, device, input, axis, dtype):
    try:
        a_np = input.astype(dtype)
    except (ValueError, OverflowError):
        return xp.zeros(())  # invalid combination of data and dtype

    a = xp.array(a_np)
    if is_module:
        return xp.max(a, axis)
    else:
        return a.max(axis)
