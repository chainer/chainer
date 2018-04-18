import math

import numpy
import pytest

import xchainer
import xchainer.testing


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


def _check_device(a, device=None):
    if device is None:
        device = xchainer.get_default_device()
    elif isinstance(device, str):
        device = xchainer.get_device(device)
    assert a.device is device


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.parametrize_dtype_specifier('dtype_spec', with_xchainer_dtypes=False)
def test_array_from_python_list_with_dtype(xp, dtype_spec, device):
    return xp.array([0, 1, 2], dtype_spec)


# TODO(sonots): Determine dtype (bool or int64, or float64) seeing values of list.
# TODO(sonots): Support nested list
@pytest.mark.parametrize('dtype', [xchainer.float64])
def test_array_from_python_list_without_dtype(dtype):
    a = xchainer.array([0, 1, 2])
    assert a.shape == (3,)
    assert a.dtype == dtype
    assert a._debug_flat_data == [0, 1, 2]


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_array_from_python_list_with_dtype_with_device(device):
    a = xchainer.array([0, 1, 2], 'float32', device)
    b = xchainer.array([0, 1, 2], 'float32')
    xchainer.testing.assert_array_equal(a, b)
    _check_device(a, device)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_array_from_numpy_ndarray(xp, shape, dtype, device):
    return xp.array(numpy.zeros(shape, dtype))


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_array_from_numpy_ndarray_with_device(shape, device):
    np_a = numpy.zeros((2,), 'float32')
    a = xchainer.array(np_a, device)
    b = xchainer.array(np_a)
    xchainer.testing.assert_array_equal(a, b)
    _check_device(a, device)


@pytest.mark.parametrize_device(['native:0'])
def test_array_from_xchainer_array(shape, dtype, device):
    t = xchainer.zeros(shape, dtype, 'native:1')
    a = xchainer.array(t)
    assert t is not a
    xchainer.testing.assert_array_equal(a, t)
    assert a.device is t.device


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_array_from_xchainer_array_with_device(device):
    shape = (2,)
    dtype = xchainer.float32
    t = xchainer.zeros(shape, dtype, 'native:0')
    a = xchainer.array(t, device)
    b = xchainer.array(t)
    assert t is not a
    xchainer.testing.assert_array_equal(a, b)
    _check_device(a, device)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.parametrize_dtype_specifier('dtype_spec')
def test_empty(xp, shape, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, xchainer.dtype):
        dtype_spec = dtype_spec.name
    a = xp.empty(shape, dtype_spec)
    a.fill(0)
    return a


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_empty_with_device(device):
    a = xchainer.empty((2,), 'float32', device)
    b = xchainer.empty((2,), 'float32')
    _check_device(a, device)
    assert a.dtype == b.dtype
    assert a.shape == b.shape


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_empty_like(xp, shape, dtype, device):
    t = xp.empty(shape, dtype)
    a = xp.empty_like(t)
    a.fill(0)
    return a


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_empty_like_with_device(device):
    t = xchainer.empty((2,), 'float32')
    a = xchainer.empty_like(t, device)
    b = xchainer.empty_like(t)
    _check_device(a, device)
    assert a.dtype == b.dtype
    assert a.shape == b.shape


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.parametrize_dtype_specifier('dtype_spec')
def test_zeros(xp, shape, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, xchainer.dtype):
        dtype_spec = dtype_spec.name
    return xp.zeros(shape, dtype_spec)


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_zeros_with_device(device):
    a = xchainer.zeros((2,), 'float32', device)
    b = xchainer.zeros((2,), 'float32')
    xchainer.testing.assert_array_equal(a, b)
    _check_device(a, device)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_zeros_like(xp, shape, dtype, device):
    t = xp.empty(shape, dtype)
    return xp.zeros_like(t)


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_zeros_like_with_device(device):
    t = xchainer.empty((2,), 'float32')
    a = xchainer.zeros_like(t, device)
    b = xchainer.zeros_like(t)
    _check_device(a, device)
    xchainer.testing.assert_array_equal(a, b)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.parametrize_dtype_specifier('dtype_spec')
def test_ones(xp, shape, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, xchainer.dtype):
        dtype_spec = dtype_spec.name
    return xp.ones(shape, dtype_spec)


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_ones_with_device(device):
    a = xchainer.ones((2,), 'float32', device)
    b = xchainer.ones((2,), 'float32')
    _check_device(a, device)
    xchainer.testing.assert_array_equal(a, b)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_ones_like(xp, shape, dtype, device):
    t = xp.empty(shape, dtype)
    return xp.ones_like(t)


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_ones_like_with_device(shape, device):
    t = xchainer.empty((2,), 'float32')
    a = xchainer.ones_like(t, device)
    b = xchainer.ones_like(t)
    _check_device(a, device)
    xchainer.testing.assert_array_equal(a, b)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('value', [True, False, -2, 0, 1, 2, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_full(xp, shape, value, device):
    return xp.full(shape, value)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('value', [True, False, -2, 0, 1, 2, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.parametrize_dtype_specifier('dtype_spec')
def test_full_with_dtype(xp, shape, dtype_spec, value, device):
    if xp is numpy and isinstance(dtype_spec, xchainer.dtype):
        dtype_spec = dtype_spec.name
    return xp.full(shape, value, dtype_spec)


@pytest.mark.parametrize('value', [True, False, -2, 0, 1, 2, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_full_with_scalar(shape, dtype, value, device):
    scalar = xchainer.Scalar(value, dtype)
    a = xchainer.full(shape, scalar)
    if scalar.dtype in (xchainer.float32, xchainer.float64) and math.isnan(float(scalar)):
        assert all([math.isnan(el) for el in a._debug_flat_data])
    else:
        assert a._debug_flat_data == [scalar.tolist()] * a.total_size


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_full_with_device(device):
    a = xchainer.full((2,), 1, 'float32', device)
    b = xchainer.full((2,), 1, 'float32')
    _check_device(a, device)
    xchainer.testing.assert_array_equal(a, b)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('value', [True, False, -2, 0, 1, 2, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_full_like(xp, shape, dtype, value, device):
    t = xp.empty(shape, dtype)
    return xp.full_like(t, value)


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_full_like_with_device(device):
    t = xchainer.empty((2,), 'float32')
    a = xchainer.full_like(t, 1, device)
    b = xchainer.full_like(t, 1)
    _check_device(a, device)
    xchainer.testing.assert_array_equal(a, b)


def _is_bool_spec(dtype_spec):
    # Used in arange tests
    if dtype_spec is None:
        return False
    return xchainer.dtype(dtype_spec) == xchainer.bool_


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('stop', [-2, 0, 0.1, 3, 3.2, False, True])
@pytest.mark.parametrize_device(['native:0'])
@xchainer.testing.parametrize_dtype_specifier('dtype_spec', additional_args=(None,))
def test_arange_stop(xp, stop, dtype_spec, device):
    # TODO(hvy): xp.arange(True) should return an ndarray of type int64
    if xp is numpy and isinstance(dtype_spec, xchainer.dtype):
        dtype_spec = dtype_spec.name
    if _is_bool_spec(dtype_spec) and stop > 2:  # Checked in test_invalid_arange_too_long_bool
        return xp.array([])
    if isinstance(stop, bool) and dtype_spec is None:
        # TODO(niboshi): This pattern needs dtype promotion.
        return xp.array([])
    return xp.arange(stop, dtype=dtype_spec)


@xchainer.testing.numpy_xchainer_array_equal()
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
@xchainer.testing.parametrize_dtype_specifier('dtype_spec', additional_args=(None,))
def test_arange_start_stop(xp, start, stop, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, xchainer.dtype):
        dtype_spec = dtype_spec.name
    if _is_bool_spec(dtype_spec) and abs(stop - start) > 2:  # Checked in test_invalid_arange_too_long_bool
        return xp.array([])
    if (isinstance(start, bool) or isinstance(stop, bool)) and dtype_spec is None:
        # TODO(niboshi): This pattern needs dtype promotion.
        return xp.array([])
    return xp.arange(start, stop, dtype=dtype_spec)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('start,stop,step', [
    (0, 3, 1),
    (0, 0, 2),
    (0, 1, 2),
    (3, -1, -2),
    (-1, 3, -2),
    (3., 2., 1.2),
    (2., -1., 1.),
    (1, 4, -1.2),
    #(4, 1, -1.2),  # TODO(niboshi): Fix it (or maybe NumPy bug?)
    (False, True, True),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.parametrize_dtype_specifier('dtype_spec', additional_args=(None,))
def test_arange_start_stop_step(xp, start, stop, step, dtype_spec, device):
    if xp is numpy and isinstance(dtype_spec, xchainer.dtype):
        dtype_spec = dtype_spec.name
    if _is_bool_spec(dtype_spec) and abs((stop - start) / step) > 2:  # Checked in test_invalid_arange_too_long_bool
        return xp.array([])
    if (isinstance(start, bool) or isinstance(stop, bool) or isinstance(step, bool)) and dtype_spec is None:
        # TODO(niboshi): This pattern needs dtype promotion.
        return xp.array([])
    return xp.arange(start, stop, step, dtype=dtype_spec)


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_arange_with_device(device):
    def check(*args, **kwargs):
        a = xchainer.arange(*args, device=device, **kwargs)
        b = xchainer.arange(*args, **kwargs)
        _check_device(a, device)
        xchainer.testing.assert_array_equal(a, b)

    check(3)
    check(3, dtype='float32')
    check(0, 3)
    check(0, 3, dtype='float32')
    check(0, 3, 2)
    check(0, 3, 2, dtype='float32')


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_invalid_arange_too_long_bool(device):
    def check(xp, err):
        with pytest.raises(err):
            xp.arange(3, dtype='bool_')
        with pytest.raises(err):
            xp.arange(1, 4, 1, dtype='bool_')
        # Should not raise since the size is <= 2.
        xp.arange(1, 4, 2, dtype='bool_')

    check(xchainer, xchainer.DtypeError)
    check(numpy, ValueError)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_invalid_arange_zero_step(device):
    def check(xp, err):
        with pytest.raises(err):
            xp.arange(1, 3, 0)

    check(xchainer, xchainer.XchainerError)
    check(numpy, ZeroDivisionError)
