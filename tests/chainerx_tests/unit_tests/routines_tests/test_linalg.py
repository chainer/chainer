import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('a_shape,b_shape', [
    ((), ()),
    ((), (2, 3)),
    ((2, 0), (0, 3)),
    ((0, 0), (0, 0)),
    ((2, 3), (3, 4)),
    # TODO(niboshi): Add test cases for more than 2 ndim
])
@pytest.mark.parametrize(
    'dtypes,chx_expected_dtype', dtype_utils.result_dtypes_two_arrays)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_dot(
        is_module, xp, device, a_shape, b_shape, dtypes, chx_expected_dtype):
    # TODO(beam2d): Remove the skip after supporting non-float dot on CUDA
    if (device.name == 'cuda:0'
            and any(numpy.dtype(dtype).kind != 'f' for dtype in dtypes)):
        pytest.skip('CUDA dot only supports floating kind dtypes.')

    a_dtype, b_dtype = dtypes

    a = array_utils.create_dummy_ndarray(xp, a_shape, a_dtype)
    b = array_utils.create_dummy_ndarray(xp, b_shape, b_dtype)

    if is_module:
        y = xp.dot(a, b)
    else:
        y = a.dot(b)

    return dtype_utils.cast_if_numpy_array(xp, y, chx_expected_dtype)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(chainerx.DimensionError, ValueError))
@pytest.mark.parametrize('a_shape,b_shape', [
    ((3, 2), (1, 3)),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_dot_invalid(is_module, xp, device, a_shape, b_shape, dtype):
    # TODO(beam2d): Remove the skip after supporting non-float dot on CUDA
    if device.name == 'cuda:0' and numpy.dtype(dtype).kind != 'f':
        return chainerx.testing.ignore()
    a = array_utils.create_dummy_ndarray(xp, a_shape, dtype)
    b = array_utils.create_dummy_ndarray(xp, b_shape, dtype)
    if is_module:
        return xp.dot(a, b)
    else:
        return a.dot(b)
