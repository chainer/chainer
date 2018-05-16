import numpy
import pytest

import xchainer
import xchainer.testing

from tests import array_utils


_shapes = [
    (),
    (0,),
    (1,),
    (2, 3),
    (1, 1, 1),
    (2, 0, 3),
]


@pytest.fixture(params=_shapes)
def shape(request):
    return request.param


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_negative(xp, device, shape, dtype, is_module):
    if dtype == 'bool_':  # Checked in test_invalid_bool_neg
        return xchainer.testing.ignore()
    x = array_utils.create_dummy_ndarray(xp, shape, dtype)
    if is_module:
        return xp.negative(x)
    else:
        return -x


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal(accept_error=(xchainer.DtypeError, TypeError))
def test_invalid_bool_negative(xp, device, is_module):
    x = xp.array([True, False], dtype='bool_')
    if is_module:
        xp.negative(x)
    else:
        -x


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_add(xp, device, shape, dtype, is_module):
    lhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=2)
    if is_module:
        return xp.add(lhs, rhs)
    else:
        return lhs + rhs


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_iadd(xp, device, shape, dtype):
    lhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=2)
    lhs += rhs
    return lhs


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_sub(xp, device, shape, numeric_dtype, is_module):
    lhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype, pattern=2)
    if is_module:
        return xp.subtract(lhs, rhs)
    else:
        return lhs - rhs


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_isub(xp, device, shape, numeric_dtype):
    lhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype, pattern=2)
    lhs -= rhs
    return lhs


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_mul(xp, device, shape, dtype, is_module):
    lhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=2)
    if is_module:
        return xp.multiply(lhs, rhs)
    else:
        return lhs * rhs


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_imul(xp, device, shape, dtype):
    lhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=2)
    lhs *= rhs
    return lhs


@pytest.mark.parametrize('scalar', [0, -1, 1, 2])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_mul_scalar(scalar, device, shape, dtype):
    x_np = array_utils.create_dummy_ndarray(numpy, shape, dtype)
    # Implicit casting in NumPy's multiply depends on the 'casting' argument,
    # which is not yet supported (xChainer always casts).
    # Therefore, we explicitly cast the scalar to the dtype of the ndarray
    # before the multiplication for NumPy.
    expected = x_np * numpy.dtype(dtype).type(scalar)

    x = xchainer.array(x_np)
    scalar_xc = xchainer.Scalar(scalar, dtype)
    xchainer.testing.assert_array_equal_ex(x * scalar, expected)
    xchainer.testing.assert_array_equal_ex(x * scalar_xc, expected)
    xchainer.testing.assert_array_equal_ex(scalar * x, expected)
    xchainer.testing.assert_array_equal_ex(scalar_xc * x, expected)
    xchainer.testing.assert_array_equal_ex(xchainer.multiply(x, scalar), expected)
    xchainer.testing.assert_array_equal_ex(xchainer.multiply(x, scalar_xc), expected)
    xchainer.testing.assert_array_equal_ex(xchainer.multiply(scalar, x), expected)
    xchainer.testing.assert_array_equal_ex(xchainer.multiply(scalar_xc, x), expected)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_truediv(xp, device, shape, numeric_dtype, is_module):
    lhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype)
    rhs = xp.arange(1, lhs.size + 1, dtype=numeric_dtype).reshape(shape)
    # TODO(beam2d): Remove astype after supporting correct dtype promotion.
    if is_module:
        return xp.divide(lhs, rhs).astype(numeric_dtype)
    else:
        return (lhs / rhs).astype(numeric_dtype)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_itruediv(xp, device, shape, numeric_dtype):
    lhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype)
    rhs = xp.arange(1, lhs.size + 1, dtype=numeric_dtype).reshape(shape)
    # TODO(beam2d): Fix after supporting correct dtype promotion.
    if xp is numpy and 'int' in numeric_dtype:
        # NumPy does not support itruediv to integer arrays.
        lhs = (lhs / rhs).astype(numeric_dtype)
    else:
        lhs /= rhs
    return lhs
