from io import StringIO
import math
import sys
import tempfile

import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils


_array_params_list = [
    -2,
    1,
    -1.5,
    2.3,
    True,
    False,
    numpy.array(1),
    float('inf'),
    float('nan'),
]


def _array_params(list):
    return list + [
        list,
        [list, list],
        (list, list),
        tuple(list),
        (tuple(list), tuple(list)),
        [tuple(list), tuple(list)],
    ]


# Traverses the entries in `obj` recursively and returns `True` if all of the
# entries are finite numbers.
def _is_all_finite(obj):
    if isinstance(obj, (tuple, list)):
        return all(_is_all_finite(o) for o in obj)
    else:
        return numpy.isfinite(obj)


def _get_default_dtype(value):
    if isinstance(value, bool):
        return 'bool_'
    if isinstance(value, int):
        return 'int32'
    if isinstance(value, float):
        return 'float32'
    assert False


# A special parameter object used to represent an unspecified argument.
class Unspecified(object):
    pass


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('obj', _array_params(_array_params_list))
@chainerx.testing.parametrize_dtype_specifier(
    'dtype_spec', additional_args=(None, Unspecified))
def test_array_from_tuple_or_list(xp, obj, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name
    # Skip nan/inf -> integer conversion that would cause a cast error.
    if (not _is_all_finite(obj)
            and dtype_spec not in (None, Unspecified)
            and chainerx.dtype(dtype_spec).kind not in ('f', 'c')):
        return chainerx.testing.ignore()
    if dtype_spec is Unspecified:
        return xp.array(obj)
    else:
        return xp.array(obj, dtype_spec)


@pytest.mark.parametrize('obj', _array_params(_array_params_list))
@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_array_from_python_tuple_or_list_with_device(obj, device):
    a = chainerx.array(obj, 'float32', device=device)
    b = chainerx.array(obj, 'float32')
    chainerx.testing.assert_array_equal_ex(a, b)
    array_utils.check_device(a, device)


def _check_array_from_numpy_array(a_chx, a_np, device=None):
    assert a_chx.offset == 0
    array_utils.check_device(a_chx, device)

    # recovered data should be equal
    a_np_recovered = chainerx.to_numpy(a_chx)
    chainerx.testing.assert_array_equal_ex(
        a_chx, a_np_recovered, strides_check=False)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_array_from_numpy_array(xp, shape, dtype, device):
    a_np = array_utils.create_dummy_ndarray(numpy, shape, dtype)
    a_xp = xp.array(a_np)

    if xp is chainerx:
        _check_array_from_numpy_array(a_xp, a_np, device)

        # test possibly freed memory
        a_np_copy = a_np.copy()
        del a_np
        chainerx.testing.assert_array_equal_ex(
            a_xp, a_np_copy, strides_check=False)

    return a_xp


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_array_from_numpy_non_contiguous_array(xp, shape, dtype, device):
    a_np = array_utils.create_dummy_ndarray(numpy, shape, dtype, padding=True)
    a_xp = xp.array(a_np)

    if xp is chainerx:
        _check_array_from_numpy_array(a_xp, a_np, device)

        # test possibly freed memory
        a_np_copy = a_np.copy()
        del a_np
        chainerx.testing.assert_array_equal_ex(
            a_xp, a_np_copy, strides_check=False)

    return a_xp


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_array_from_numpy_positive_offset_array(xp, device):
    a_np = array_utils.create_dummy_ndarray(numpy, (2, 3), 'int32')[1, 1:]
    a_xp = xp.array(a_np)

    if xp is chainerx:
        _check_array_from_numpy_array(a_xp, a_np, device)

        # test possibly freed memory
        a_np_copy = a_np.copy()
        del a_np
        chainerx.testing.assert_array_equal_ex(a_xp, a_np_copy)

    return a_xp


def _array_from_numpy_array_with_dtype(xp, shape, src_dtype, dst_dtype_spec):
    if xp is numpy and isinstance(dst_dtype_spec, chainerx.dtype):
        dst_dtype_spec = dst_dtype_spec.name
    t = array_utils.create_dummy_ndarray(numpy, shape, src_dtype)
    return xp.array(t, dtype=dst_dtype_spec)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('src_dtype', chainerx.testing.all_dtypes)
@pytest.mark.parametrize('dst_dtype', chainerx.testing.all_dtypes)
def test_array_from_numpy_array_with_dtype(
        xp, shape, src_dtype, dst_dtype, device):
    return _array_from_numpy_array_with_dtype(xp, shape, src_dtype, dst_dtype)


@chainerx.testing.numpy_chainerx_array_equal()
@chainerx.testing.parametrize_dtype_specifier(
    'dst_dtype_spec', additional_args=(None,))
def test_array_from_numpy_array_with_dtype_spec(xp, shape, dst_dtype_spec):
    return _array_from_numpy_array_with_dtype(
        xp, shape, 'float32', dst_dtype_spec)


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_array_from_numpy_array_with_device(shape, device):
    orig = array_utils.create_dummy_ndarray(numpy, (2, ), 'float32')
    a = chainerx.array(orig, device=device)
    b = chainerx.array(orig)
    chainerx.testing.assert_array_equal_ex(a, b)
    array_utils.check_device(a, device)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('copy', [True, False])
def test_array_from_chainerx_array(shape, dtype, copy, device):
    t = array_utils.create_dummy_ndarray(chainerx, shape, dtype, device=device)
    a = chainerx.array(t, copy=copy)
    if not copy:
        assert t is a
    else:
        assert t is not a
        chainerx.testing.assert_array_equal_ex(a, t, strides_check=False)
        assert a.device is t.device
        assert a.is_contiguous


def _check_array_from_chainerx_array_with_dtype(
        shape, src_dtype, dst_dtype_spec, copy, device=None):
    t = array_utils.create_dummy_ndarray(
        chainerx, shape, src_dtype, device=device)
    a = chainerx.array(t, dtype=dst_dtype_spec, copy=copy)

    src_dtype = chainerx.dtype(src_dtype)
    dst_dtype = src_dtype if dst_dtype_spec is None else chainerx.dtype(
        dst_dtype_spec)
    device = chainerx.get_device(device)

    if (not copy
            and src_dtype == dst_dtype
            and device is chainerx.get_default_device()):
        assert t is a
    else:
        assert t is not a
        chainerx.testing.assert_array_equal_ex(a, t.astype(dst_dtype))
        assert a.dtype == dst_dtype
        assert a.device is chainerx.get_default_device()


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('src_dtype', chainerx.testing.all_dtypes)
@pytest.mark.parametrize('dst_dtype', chainerx.testing.all_dtypes)
@pytest.mark.parametrize('copy', [True, False])
def test_array_from_chainerx_array_with_dtype(
        shape, src_dtype, dst_dtype, copy, device):
    _check_array_from_chainerx_array_with_dtype(
        shape, src_dtype, dst_dtype, copy, device)


@chainerx.testing.parametrize_dtype_specifier(
    'dst_dtype_spec', additional_args=(None,))
@pytest.mark.parametrize('copy', [True, False])
def test_array_from_chainerx_array_with_dtype_spec(
        shape, dst_dtype_spec, copy):
    _check_array_from_chainerx_array_with_dtype(
        shape, 'float32', dst_dtype_spec, copy)


@pytest.mark.parametrize('src_dtype', chainerx.testing.all_dtypes)
@pytest.mark.parametrize('dst_dtype', chainerx.testing.all_dtypes)
@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize(
    'dst_device_spec',
    [None, 'native:1', chainerx.get_device('native:1'), 'native:0'])
def test_array_from_chainerx_array_with_device(
        src_dtype, dst_dtype, copy, device, dst_device_spec):
    t = array_utils.create_dummy_ndarray(
        chainerx, (2,), src_dtype, device=device)
    a = chainerx.array(t, dtype=dst_dtype, copy=copy, device=dst_device_spec)

    dst_device = chainerx.get_device(dst_device_spec)

    if not copy and src_dtype == dst_dtype and device is dst_device:
        assert t is a
    else:
        assert t is not a
        chainerx.testing.assert_array_equal_ex(
            a, t.to_device(dst_device).astype(dst_dtype))
        assert a.dtype == chainerx.dtype(dst_dtype)
        assert a.device is dst_device


def test_asarray_from_python_tuple_or_list():
    obj = _array_params_list
    a = chainerx.asarray(obj, dtype='float32')
    e = chainerx.array(obj, dtype='float32', copy=False)
    chainerx.testing.assert_array_equal_ex(e, a)
    assert e.device is a.device


def test_asarray_from_numpy_array_with_zero_copy():
    obj = array_utils.create_dummy_ndarray(
        numpy, (2, 3), 'float32', padding=True)
    obj_refcount_before = sys.getrefcount(obj)

    a = chainerx.asarray(obj, dtype='float32')

    assert sys.getrefcount(obj) == obj_refcount_before + 1
    chainerx.testing.assert_array_equal_ex(obj, a)

    # test buffer is shared (zero copy)
    a += a
    chainerx.testing.assert_array_equal_ex(obj, a)

    # test possibly freed memory
    obj_copy = obj.copy()
    del obj
    chainerx.testing.assert_array_equal_ex(obj_copy, a, strides_check=False)

    # test possibly freed memory (the other way)
    obj = array_utils.create_dummy_ndarray(
        numpy, (2, 3), 'float32', padding=True)
    a = chainerx.asarray(obj, dtype='float32')
    a_copy = a.copy()
    del a
    chainerx.testing.assert_array_equal_ex(a_copy, obj, strides_check=False)


def test_asarray_from_numpy_array_with_copy():
    obj = array_utils.create_dummy_ndarray(numpy, (2, 3), 'int32')
    a = chainerx.asarray(obj, dtype='float32')
    e = chainerx.array(obj, dtype='float32', copy=False)
    chainerx.testing.assert_array_equal_ex(e, a)
    assert e.device is a.device

    # test buffer is not shared
    a += a
    assert not numpy.array_equal(obj, chainerx.to_numpy(a))


@pytest.mark.parametrize('dtype', ['int32', 'float32'])
def test_asarray_from_chainerx_array(dtype):
    obj = array_utils.create_dummy_ndarray(chainerx, (2, 3), 'int32')
    a = chainerx.asarray(obj, dtype=dtype)
    if a.dtype == obj.dtype:
        assert a is obj
    else:
        assert a is not obj
    e = chainerx.array(obj, dtype=dtype, copy=False)
    chainerx.testing.assert_array_equal_ex(e, a)
    assert e.device is a.device


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_asarray_with_device(device):
    a = chainerx.asarray([0, 1], 'float32', device)
    b = chainerx.asarray([0, 1], 'float32')
    chainerx.testing.assert_array_equal_ex(a, b)
    array_utils.check_device(a, device)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('padding', [False, True])
def test_ascontiguousarray_from_numpy_array(xp, shape, dtype, padding):
    obj = array_utils.create_dummy_ndarray(
        numpy, shape, dtype, padding=padding)
    a = xp.ascontiguousarray(obj)
    if xp is chainerx:
        assert a.is_contiguous
    return a


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('obj', _array_params(_array_params_list))
@chainerx.testing.parametrize_dtype_specifier(
    'dtype_spec', additional_args=(None, Unspecified))
def test_ascontiguousarray_from_tuple_or_list(xp, device, obj, dtype_spec):
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name
    # Skip nan/inf -> integer conversion that would cause a cast error.
    if (not _is_all_finite(obj)
            and dtype_spec not in (None, Unspecified)
            and chainerx.dtype(dtype_spec).kind not in ('f', 'c')):
        return chainerx.testing.ignore()

    if dtype_spec is Unspecified:
        a = xp.ascontiguousarray(obj)
    else:
        a = xp.ascontiguousarray(obj, dtype=dtype_spec)

    if xp is chainerx:
        assert a.is_contiguous
    return a


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('padding', [False, True])
def test_ascontiguousarray_from_chainerx_array(device, shape, dtype, padding):
    np_arr = array_utils.create_dummy_ndarray(
        numpy, shape, dtype, padding=padding)
    obj = chainerx.testing._fromnumpy(np_arr, keepstrides=True, device=device)
    a = chainerx.ascontiguousarray(obj)
    if not padding and shape != ():  # () will be reshaped to (1,)
        assert a is obj
    e = chainerx.ascontiguousarray(np_arr)
    chainerx.testing.assert_array_equal_ex(e, a, strides_check=False)
    assert a.is_contiguous
    assert e.dtype.name == a.dtype.name


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('padding', [False, True])
@chainerx.testing.parametrize_dtype_specifier('dtype_spec')
def test_ascontiguousarray_with_dtype(xp, device, shape, padding, dtype_spec):
    obj = array_utils.create_dummy_ndarray(xp, shape, 'int32', padding=padding)
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name
    a = xp.ascontiguousarray(obj, dtype=dtype_spec)
    if xp is chainerx:
        assert a.is_contiguous
    return a


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1'), 'native:0'])
@pytest.mark.parametrize('padding', [False, True])
def test_ascontiguousarray_with_device(device, shape, padding, dtype):
    obj = array_utils.create_dummy_ndarray(
        chainerx, shape, dtype, padding=padding)
    a = chainerx.ascontiguousarray(obj, device=device)
    b = chainerx.ascontiguousarray(obj)
    array_utils.check_device(a, device)
    assert a.is_contiguous
    chainerx.testing.assert_array_equal_ex(a, b)


def test_asanyarray_from_python_tuple_or_list():
    obj = _array_params_list
    a = chainerx.asanyarray(obj, dtype='float32')
    e = chainerx.array(obj, dtype='float32', copy=False)
    chainerx.testing.assert_array_equal_ex(e, a)
    assert e.device is a.device


def test_asanyarray_from_numpy_array():
    obj = array_utils.create_dummy_ndarray(numpy, (2, 3), 'int32')
    a = chainerx.asanyarray(obj, dtype='float32')
    e = chainerx.array(obj, dtype='float32', copy=False)
    chainerx.testing.assert_array_equal_ex(e, a)
    assert e.device is a.device


def test_asanyarray_from_numpy_subclass_array():
    class Subclass(numpy.ndarray):
        pass
    obj = array_utils.create_dummy_ndarray(
        numpy, (2, 3), 'int32').view(Subclass)
    a = chainerx.asanyarray(obj, dtype='float32')
    e = chainerx.array(obj, dtype='float32', copy=False)
    chainerx.testing.assert_array_equal_ex(e, a)
    assert e.device is a.device


@pytest.mark.parametrize('dtype', ['int32', 'float32'])
def test_asanyarray_from_chainerx_array(dtype):
    obj = array_utils.create_dummy_ndarray(chainerx, (2, 3), 'int32')
    a = chainerx.asanyarray(obj, dtype=dtype)
    if a.dtype == obj.dtype:
        assert a is obj
    else:
        assert a is not obj
    e = chainerx.array(obj, dtype=dtype, copy=False)
    chainerx.testing.assert_array_equal_ex(e, a)
    assert e.device is a.device


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_asanyarray_with_device(device):
    a = chainerx.asanyarray([0, 1], 'float32', device)
    b = chainerx.asanyarray([0, 1], 'float32')
    chainerx.testing.assert_array_equal_ex(a, b)
    array_utils.check_device(a, device)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@chainerx.testing.parametrize_dtype_specifier(
    'dtype_spec', additional_args=(None, Unspecified))
def test_empty(xp, shape_as_tuple_or_int, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name
    if dtype_spec is Unspecified:
        a = xp.empty(shape_as_tuple_or_int)
    else:
        a = xp.empty(shape_as_tuple_or_int, dtype_spec)
    a.fill(0)
    if dtype_spec in (None, Unspecified):
        a = dtype_utils.cast_if_numpy_array(xp, a, 'float32')
    return a


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_empty_with_device(device):
    a = chainerx.empty((2,), 'float32', device)
    b = chainerx.empty((2,), 'float32')
    array_utils.check_device(a, device)
    assert a.dtype == b.dtype
    assert a.shape == b.shape


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_empty_like(xp, shape, dtype, device):
    t = xp.empty(shape, dtype)
    a = xp.empty_like(t)
    a.fill(0)
    return a


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_empty_like_with_device(device):
    t = chainerx.empty((2,), 'float32')
    a = chainerx.empty_like(t, device)
    b = chainerx.empty_like(t)
    array_utils.check_device(a, device)
    assert a.dtype == b.dtype
    assert a.shape == b.shape


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@chainerx.testing.parametrize_dtype_specifier(
    'dtype_spec', additional_args=(None, Unspecified))
def test_zeros(xp, shape_as_tuple_or_int, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name
    if dtype_spec is Unspecified:
        out = xp.zeros(shape_as_tuple_or_int)
    else:
        out = xp.zeros(shape_as_tuple_or_int, dtype_spec)
    if dtype_spec in (None, Unspecified):
        out = dtype_utils.cast_if_numpy_array(xp, out, 'float32')
    return out


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_zeros_with_device(device):
    a = chainerx.zeros((2,), 'float32', device)
    b = chainerx.zeros((2,), 'float32')
    chainerx.testing.assert_array_equal_ex(a, b)
    array_utils.check_device(a, device)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_zeros_like(xp, shape, dtype, device):
    t = xp.empty(shape, dtype)
    return xp.zeros_like(t)


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_zeros_like_with_device(device):
    t = chainerx.empty((2,), 'float32')
    a = chainerx.zeros_like(t, device)
    b = chainerx.zeros_like(t)
    array_utils.check_device(a, device)
    chainerx.testing.assert_array_equal_ex(a, b)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@chainerx.testing.parametrize_dtype_specifier(
    'dtype_spec', additional_args=(None, Unspecified))
def test_ones(xp, shape_as_tuple_or_int, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name
    if dtype_spec is Unspecified:
        out = xp.ones(shape_as_tuple_or_int)
    else:
        out = xp.ones(shape_as_tuple_or_int, dtype_spec)
    if dtype_spec in (None, Unspecified):
        out = dtype_utils.cast_if_numpy_array(xp, out, 'float32')
    return out


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_ones_with_device(device):
    a = chainerx.ones((2,), 'float32', device)
    b = chainerx.ones((2,), 'float32')
    array_utils.check_device(a, device)
    chainerx.testing.assert_array_equal_ex(a, b)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_ones_like(xp, shape, dtype, device):
    t = xp.empty(shape, dtype)
    return xp.ones_like(t)


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_ones_like_with_device(shape, device):
    t = chainerx.empty((2,), 'float32')
    a = chainerx.ones_like(t, device)
    b = chainerx.ones_like(t)
    array_utils.check_device(a, device)
    chainerx.testing.assert_array_equal_ex(a, b)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize(
    'value', [True, False, -2, 0, 1, 2, 2.3, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_full(xp, shape_as_tuple_or_int, value, device):
    out = xp.full(shape_as_tuple_or_int, value)
    return dtype_utils.cast_if_numpy_array(xp, out, _get_default_dtype(value))


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize(
    'value', [True, False, -2, 0, 1, 2, 2.3, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@chainerx.testing.parametrize_dtype_specifier('dtype_spec')
def test_full_with_dtype(xp, shape, dtype_spec, value, device):
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name
    return xp.full(shape, value, dtype_spec)


@pytest.mark.parametrize(
    'value', [True, False, -2, 0, 1, 2, 2.5, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_full_with_scalar(shape, dtype, value, device):
    scalar = chainerx.Scalar(value)
    a = chainerx.full(shape, scalar)
    if isinstance(value, float) and math.isnan(value):
        assert all([math.isnan(el) for el in a._debug_flat_data])
    else:
        assert a._debug_flat_data == [scalar.tolist()] * a.size


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_full_with_device(device):
    a = chainerx.full((2,), 1, 'float32', device)
    b = chainerx.full((2,), 1, 'float32')
    array_utils.check_device(a, device)
    chainerx.testing.assert_array_equal_ex(a, b)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize(
    'value', [True, False, -2, 0, 1, 2, 2.3, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_full_like(xp, shape, dtype, value, device):
    t = xp.empty(shape, dtype)
    return xp.full_like(t, value)


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_full_like_with_device(device):
    t = chainerx.empty((2,), 'float32')
    a = chainerx.full_like(t, 1, device)
    b = chainerx.full_like(t, 1)
    array_utils.check_device(a, device)
    chainerx.testing.assert_array_equal_ex(a, b)


def _is_bool_spec(dtype_spec):
    # Used in arange tests
    if dtype_spec is None:
        return False
    return chainerx.dtype(dtype_spec) == chainerx.bool_


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('stop', [-2, 0, 0.1, 3, 3.2, False, True])
@pytest.mark.parametrize_device(['native:0'])
@chainerx.testing.parametrize_dtype_specifier(
    'dtype_spec', additional_args=(None,))
def test_arange_stop(xp, stop, dtype_spec, device):
    # TODO(hvy): xp.arange(True) should return an ndarray of type int64
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name
    # Checked in test_invalid_arange_too_long_bool
    if _is_bool_spec(dtype_spec) and stop > 2:
        return chainerx.testing.ignore()
    if isinstance(stop, bool) and dtype_spec is None:
        # TODO(niboshi): This pattern needs dtype promotion.
        return chainerx.testing.ignore()
    out = xp.arange(stop, dtype=dtype_spec)
    if dtype_spec in (None, Unspecified):
        expected_dtype = _get_default_dtype(stop)
        out = dtype_utils.cast_if_numpy_array(xp, out, expected_dtype)
    return out


@chainerx.testing.numpy_chainerx_allclose(
    atol=1e-7, float16_rtol=5e-3, float16_atol=5e-3)
@pytest.mark.parametrize('start,stop', [
    (0, 0),
    (0, 3),
    (-3, 2),
    (2, 0),
    (-2.2, 3.4),
    (True, True),
    (False, False),
    (True, False),
    (False, True),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@chainerx.testing.parametrize_dtype_specifier(
    'dtype_spec', additional_args=(None,))
def test_arange_start_stop(xp, start, stop, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name
    # Checked in test_invalid_arange_too_long_bool
    if _is_bool_spec(dtype_spec) and abs(stop - start) > 2:
        return chainerx.testing.ignore()
    if ((isinstance(start, bool)
            or isinstance(stop, bool))
            and dtype_spec is None):
        # TODO(niboshi): This pattern needs dtype promotion.
        return chainerx.testing.ignore()
    out = xp.arange(start, stop, dtype=dtype_spec)
    if dtype_spec in (None, Unspecified):
        expected_dtype = _get_default_dtype(stop)
        out = dtype_utils.cast_if_numpy_array(xp, out, expected_dtype)
    return out


@chainerx.testing.numpy_chainerx_allclose(float16_rtol=1e-3)
@pytest.mark.parametrize('start,stop,step', [
    (0, 3, 1),
    (0, 0, 2),
    (0, 1, 2),
    (3, -1, -2),
    (-1, 3, -2),
    (3., 2., 1.2),
    (2., -1., 1.),
    (1, 4, -1.2),
    # (4, 1, -1.2),  # TODO(niboshi): Fix it (or maybe NumPy bug?)
    (False, True, True),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@chainerx.testing.parametrize_dtype_specifier(
    'dtype_spec', additional_args=(None,))
def test_arange_start_stop_step(xp, start, stop, step, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name
    # Checked in test_invalid_arange_too_long_bool
    if _is_bool_spec(dtype_spec) and abs((stop - start) / step) > 2:
        return chainerx.testing.ignore()
    if ((isinstance(start, bool)
            or isinstance(stop, bool)
            or isinstance(step, bool))
            and dtype_spec is None):
        # TODO(niboshi): This pattern needs dtype promotion.
        return chainerx.testing.ignore()
    out = xp.arange(start, stop, step, dtype=dtype_spec)
    if dtype_spec in (None, Unspecified):
        expected_dtype = _get_default_dtype(step)
        out = dtype_utils.cast_if_numpy_array(xp, out, expected_dtype)
    return out


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_arange_with_device(device):
    def check(*args, **kwargs):
        a = chainerx.arange(*args, device=device, **kwargs)
        b = chainerx.arange(*args, **kwargs)
        array_utils.check_device(a, device)
        chainerx.testing.assert_array_equal_ex(a, b)

    check(3)
    check(3, dtype='float32')
    check(0, 3)
    check(0, 3, dtype='float32')
    check(0, 3, 2)
    check(0, 3, 2, dtype='float32')


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_arange_invalid_too_long_bool(device):
    def check(xp, err):
        with pytest.raises(err):
            xp.arange(3, dtype='bool_')
        with pytest.raises(err):
            xp.arange(1, 4, 1, dtype='bool_')
        # Should not raise since the size is <= 2.
        xp.arange(1, 4, 2, dtype='bool_')

    check(chainerx, chainerx.DtypeError)
    check(numpy, ValueError)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_arange_invalid_zero_step(device):
    def check(xp, err):
        with pytest.raises(err):
            xp.arange(1, 3, 0)

    check(chainerx, chainerx.ChainerxError)
    check(numpy, ZeroDivisionError)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@chainerx.testing.parametrize_dtype_specifier('dtype_spec')
@pytest.mark.parametrize('n', [0, 1, 2, 257])
def test_identity(xp, n, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name
    out = xp.identity(n, dtype_spec)
    if dtype_spec in (None, Unspecified):
        out = dtype_utils.cast_if_numpy_array(xp, out, 'float32')
    return out


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_identity_with_device(device):
    a = chainerx.identity(3, 'float32', device)
    b = chainerx.identity(3, 'float32')
    array_utils.check_device(a, device)
    chainerx.testing.assert_array_equal_ex(a, b)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(ValueError, chainerx.DimensionError))
@pytest.mark.parametrize('device', ['native:0', 'native:0'])
def test_identity_invalid_negative_n(xp, device):
    xp.identity(-1, 'float32')


@chainerx.testing.numpy_chainerx_array_equal(accept_error=(TypeError,))
@pytest.mark.parametrize('device', ['native:1', 'native:0'])
def test_identity_invalid_n_type(xp, device):
    xp.identity(3.0, 'float32')


# TODO(hvy): Add tests with non-ndarray but array-like inputs when supported.
@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('N,M,k', [
    (0, 0, 0),
    (0, 0, 1),
    (2, 1, -2),
    (2, 1, -1),
    (2, 1, 0),
    (2, 1, 1),
    (2, 1, 2),
    (3, 4, -4),
    (3, 4, -1),
    (3, 4, 1),
    (3, 4, 4),
    (6, 3, 1),
    (6, 3, -1),
    (3, 6, 1),
    (3, 6, -1),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@chainerx.testing.parametrize_dtype_specifier('dtype_spec')
def test_eye(xp, N, M, k, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name
    out = xp.eye(N, M, k, dtype_spec)
    if dtype_spec in (None, Unspecified):
        out = dtype_utils.cast_if_numpy_array(xp, out, 'float32')
    return out


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('N,M,k', [
    (3, None, 1),
    (3, 4, None),
    (3, None, None),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@chainerx.testing.parametrize_dtype_specifier('dtype_spec')
def test_eye_with_default(xp, N, M, k, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name

    if M is None and k is None:
        return xp.eye(N, dtype=dtype_spec)
    elif M is None:
        return xp.eye(N, k=k, dtype=dtype_spec)
    elif k is None:
        return xp.eye(N, M=M, dtype=dtype_spec)
    assert False


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_eye_with_device(device):
    a = chainerx.eye(1, 2, 1, 'float32', device)
    b = chainerx.eye(1, 2, 1, 'float32')
    array_utils.check_device(a, device)
    chainerx.testing.assert_array_equal_ex(a, b)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(ValueError, chainerx.DimensionError))
@pytest.mark.parametrize('N,M', [
    (-1, 2),
    (1, -1),
    (-2, -1),
])
@pytest.mark.parametrize('device', ['native:0', 'native:0'])
def test_eye_invalid_negative_N_M(xp, N, M, device):
    xp.eye(N, M, 1, 'float32')


@chainerx.testing.numpy_chainerx_array_equal(accept_error=(TypeError,))
@pytest.mark.parametrize('N,M,k', [
    (1.0, 2, 1),
    (2, 1.0, 1),
    (2, 3, 1.0),
    (2.0, 1.0, 1),
])
@pytest.mark.parametrize('device', ['native:1', 'native:0'])
def test_eye_invalid_NMk_type(xp, N, M, k, device):
    xp.eye(N, M, k, 'float32')


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('k', [0, -2, -1, 1, 2, -5, 4])
@pytest.mark.parametrize('shape', [(4,), (2, 3), (6, 5)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_diag(xp, k, shape, transpose, device):
    v = xp.arange(array_utils.total_size(shape), dtype='int32').reshape(shape)
    if transpose:  # Test non-contiguous inputs for multi-dimensional shapes.
        v = v.T
    return xp.diag(v, k)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(ValueError, chainerx.DimensionError))
@pytest.mark.parametrize('k', [0, -2, -1, 1, 2, -5, 4])
@pytest.mark.parametrize('shape', [(), (2, 1, 2), (2, 0, 1)])
@pytest.mark.parametrize('device', ['native:1', 'native:0'])
def test_diag_invalid_ndim(xp, k, shape, device):
    v = xp.arange(array_utils.total_size(shape), dtype='int32').reshape(shape)
    return xp.diag(v, k)


# TODO(hvy): Add tests with non-ndarray but array-like inputs when supported.
@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('k', [0, -2, -1, 1, 2, -5, 4])
@pytest.mark.parametrize('shape', [(), (4,), (2, 3), (6, 5), (2, 0)])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_diagflat(xp, k, shape, device):
    v = xp.arange(array_utils.total_size(shape), dtype='int32').reshape(shape)
    return xp.diagflat(v, k)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(ValueError, chainerx.DimensionError))
@pytest.mark.parametrize('k', [0, -2, -1, 1, 2, -5, 4])
@pytest.mark.parametrize('shape', [(2, 1, 2), (2, 0, 1)])
@pytest.mark.parametrize('device', ['native:1', 'native:0'])
def test_diagflat_invalid_ndim(xp, k, shape, device):
    v = xp.arange(array_utils.total_size(shape), dtype='int32').reshape(shape)
    return xp.diagflat(v, k)


@chainerx.testing.numpy_chainerx_allclose(float16_rtol=1e-3)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('start,stop', [
    (0, 0),
    (0, 1),
    (1, 0),
    (-1, 0),
    (0, -1),
    (1, -1),
    (-13.3, 352.5),
    (13.3, -352.5),
])
@pytest.mark.parametrize('num', [0, 1, 2, 257])
@pytest.mark.parametrize('endpoint', [True, False])
@pytest.mark.parametrize('range_type', [float, int])
def test_linspace(xp, start, stop, num, endpoint, range_type, dtype, device):
    start = range_type(start)
    stop = range_type(stop)
    return xp.linspace(start, stop, num, endpoint=endpoint, dtype=dtype)


@chainerx.testing.numpy_chainerx_allclose()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@chainerx.testing.parametrize_dtype_specifier('dtype_spec')
def test_linspace_dtype_spec(xp, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name
    return xp.linspace(3, 5, 10, dtype=dtype_spec)


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_linspace_with_device(device):
    a = chainerx.linspace(3, 5, 10, dtype='float32', device=device)
    b = chainerx.linspace(3, 5, 10, dtype='float32')
    array_utils.check_device(a, device)
    chainerx.testing.assert_array_equal_ex(a, b)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(ValueError, chainerx.ChainerxError))
@pytest.mark.parametrize('device', ['native:0', 'native:0'])
def test_linspace_invalid_num(xp, device):
    xp.linspace(2, 4, -1)


@pytest.mark.parametrize_device(['native:0'])
def test_frombuffer_from_numpy_array(device):
    obj = array_utils.create_dummy_ndarray(
        numpy, (2, 3), 'int32', padding=False)

    a_chx = chainerx.frombuffer(obj, obj.dtype)
    a_np = numpy.frombuffer(obj, obj.dtype)

    chainerx.testing.assert_array_equal_ex(a_np, a_chx)
    assert a_chx.device is chainerx.get_device(device)

    # test buffer is shared
    obj += obj
    chainerx.testing.assert_array_equal_ex(obj.ravel(), a_chx)

    # test possibly freed memory
    obj_copy = obj.copy()
    del obj
    chainerx.testing.assert_array_equal_ex(obj_copy.ravel(), a_chx)


@pytest.mark.parametrize_device(['cuda:0'])
def test_frombuffer_from_numpy_array_with_cuda(device):
    obj = array_utils.create_dummy_ndarray(numpy, (2, 3), 'int32')
    with pytest.raises(chainerx.ChainerxError):
        chainerx.frombuffer(obj, obj.dtype)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(ValueError, chainerx.ChainerxError))
def test_frombuffer_from_numpy_array_with_noncontiguous(xp):
    obj = array_utils.create_dummy_ndarray(
        numpy, (2, 3), 'int32', padding=True)
    return xp.frombuffer(obj, obj.dtype)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(ValueError, chainerx.ChainerxError))
@pytest.mark.parametrize('count', [-1, 0, 1, 3, 4])
@pytest.mark.parametrize('offset', [-1, 0, 1, 4, 3 * 4, 3 * 4 + 4])
def test_frombuffer_from_numpy_array_with_offset_count(xp, count, offset):
    obj = array_utils.create_dummy_ndarray(numpy, (3,), 'int32')
    return xp.frombuffer(obj, obj.dtype, count=count, offset=offset)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_frombuffer_from_device_buffer(device):
    dtype = 'int32'

    device_buffer = chainerx.testing._DeviceBuffer(
        [1, 2, 3, 4, 5, 6], (2, 3), dtype)
    a = chainerx.frombuffer(device_buffer, dtype)
    e = chainerx.array([1, 2, 3, 4, 5, 6], dtype)

    chainerx.testing.assert_array_equal_ex(e, a)
    assert a.device is chainerx.get_device(device)


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
def test_frombuffer_with_device(device):
    obj = array_utils.create_dummy_ndarray(
        numpy, (2, 3), 'int32', padding=False)
    a = chainerx.frombuffer(obj, obj.dtype, device=device)
    b = chainerx.frombuffer(obj, obj.dtype)
    chainerx.testing.assert_array_equal_ex(a, b)
    array_utils.check_device(a, device)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('count', [-1, 0, 2])
@pytest.mark.parametrize('sep', ['', 'a'])
@pytest.mark.parametrize('device', ['native:0', 'cuda:0'])
@chainerx.testing.parametrize_dtype_specifier('dtype_spec')
def test_fromfile(xp, count, sep, dtype_spec, device):
    # Write array data to temporary file.
    if isinstance(dtype_spec, chainerx.dtype):
        numpy_dtype_spec = dtype_spec.name
    else:
        numpy_dtype_spec = dtype_spec
    data = numpy.arange(2, dtype=numpy_dtype_spec)
    f = tempfile.TemporaryFile()
    data.tofile(f, sep=sep)

    # Read file.
    f.seek(0)
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = numpy_dtype_spec
    return xp.fromfile(f, dtype=dtype_spec, count=count, sep=sep)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('device', ['native:0', 'cuda:0'])
@chainerx.testing.parametrize_dtype_specifier('dtype_spec')
def test_loadtxt(xp, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name

    txt = '''// Comment to be ignored.
1 2 3 4
5 6 7 8
'''
    txt = StringIO(txt)

    # Converter that is used to add 1 to each element in the 3rd column.
    def converter(element_str):
        return float(element_str) + 1

    return xp.loadtxt(
        txt, dtype=dtype_spec, comments='//', delimiter=' ',
        converters={3: converter}, skiprows=2, usecols=(1, 3), unpack=False,
        ndmin=2, encoding='bytes')


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('count', [-1, 0, 5])
@pytest.mark.parametrize('device', ['native:0', 'cuda:0'])
@chainerx.testing.parametrize_dtype_specifier('dtype_spec')
def test_fromiter(xp, count, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name

    iterable = (x * x for x in range(5))
    return xp.fromiter(iterable, dtype=dtype_spec, count=count)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('count', [-1, 0, 3])
@pytest.mark.parametrize('sep', [' ', 'a'])
@pytest.mark.parametrize('device', ['native:0', 'cuda:0'])
@chainerx.testing.parametrize_dtype_specifier('dtype_spec')
def test_fromstring(xp, count, sep, dtype_spec, device):
    if isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name

    elements = ['1', '2', '3']
    string = sep.join(elements)
    return xp.fromstring(string, dtype=dtype_spec, count=count, sep=sep)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('device', ['native:0', 'cuda:0'])
@chainerx.testing.parametrize_dtype_specifier('dtype_spec')
def test_fromfunction(xp, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, chainerx.dtype):
        dtype_spec = dtype_spec.name

    def function(i, j, addend):
        return i * j + addend

    # addend should be passed as a keyword argument to function.
    return xp.fromfunction(function, (2, 2), dtype=dtype_spec, addend=2)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_copy(xp, shape, dtype, device, is_module):
    a = array_utils.create_dummy_ndarray(xp, shape, dtype)
    if is_module:
        return xp.copy(a)
    else:
        return a.copy()
