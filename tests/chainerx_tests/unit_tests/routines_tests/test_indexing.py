import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils


# TODO(niboshi): Remove strides_check=False
@chainerx.testing.numpy_chainerx_array_equal(strides_check=False)
@pytest.mark.parametrize('shape,indices', [
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
def test_getitem(xp, shape, indices):
    a = array_utils.create_dummy_ndarray(xp, shape, 'int32')
    return a[indices]


# TODO(hvy): Add cases where axis=None, when supported.
# TODO(hvy): Add cases where indices is not int64, when supported.
# shape,indices,axis
_take_valid_params = [
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
]

_take_invalid_params = [
    # Axis out of bounds
    ((2, 3), [0], 2),
    ((2, 3), [0], -3),
]


@chainerx.testing.numpy_chainerx_array_equal(
    dtype_check=False, accept_error=(chainerx.DimensionError, numpy.AxisError))
@pytest.mark.parametrize(
    'shape,indices,axis', _take_valid_params + _take_invalid_params)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_take_list_indices(is_module, xp, shape, indices, axis, device):
    a = array_utils.create_dummy_ndarray(xp, shape, 'float32')

    assert isinstance(indices, list)

    if is_module:
        return xp.take(a, indices, axis)
    else:
        return a.take(indices, axis)


@chainerx.testing.numpy_chainerx_array_equal(
    dtype_check=False, accept_error=(chainerx.DimensionError, numpy.AxisError))
@pytest.mark.parametrize(
    'shape,indices,axis', _take_valid_params + _take_invalid_params)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_take_numpy_indices(is_module, xp, shape, indices, axis, device):
    a = array_utils.create_dummy_ndarray(xp, shape, 'float32')

    # dtype is cast to int64 since no other dtypes are currently supported by
    # chainerx.take
    indices = numpy.array(indices).astype('int64')

    if is_module:
        return xp.take(a, indices, axis)
    else:
        return a.take(indices, axis)


@chainerx.testing.numpy_chainerx_array_equal(
    dtype_check=False, accept_error=(chainerx.DimensionError, numpy.AxisError))
@pytest.mark.parametrize(
    'shape,indices,axis', _take_valid_params + _take_invalid_params)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_take_chainerx_indices(is_module, xp, shape, indices, axis, device):
    a = array_utils.create_dummy_ndarray(xp, shape, 'float32')

    # First convert to ndarray since some indices are nested lists which
    # chainerx cannot convert. Additionally, dtype is cast to int64 since no
    # other dtypes are currently supported by chainerx.take
    indices = numpy.array(indices).astype('int64')

    if is_module:
        return xp.take(a, xp.array(indices), axis)
    else:
        return a.take(xp.array(indices), axis)
