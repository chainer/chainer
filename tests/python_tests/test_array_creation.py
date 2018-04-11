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
def test_array_from_python_list(xp, dtype):
    return xp.array([0, 1, 2], xp.dtype(dtype.name))


# TODO(sonots): Determine dtype (bool or int64, or float64) seeing values of list.
# TODO(sonots): Support nested list
@pytest.mark.parametrize('dtype', [xchainer.float64])
def test_array_from_python_list_without_dtype(dtype):
    a = xchainer.array([0, 1, 2])
    assert a.shape == (3,)
    assert a.dtype == dtype
    assert a._debug_flat_data == [0, 1, 2]


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_array_from_python_list_with_device(device):
    a = xchainer.array([0, 1, 2], 'f', device)
    _check_device(a, device)


@xchainer.testing.numpy_xchainer_array_equal()
def test_array_from_numpy_ndarray(xp, shape, dtype):
    return xp.array(numpy.zeros(shape, numpy.dtype(dtype.name)))


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_array_from_numpy_ndarray_with_device(shape, dtype, device):
    a = xchainer.array(numpy.zeros((2,), 'f'), device)
    _check_device(a, device)


@pytest.mark.parametrize_device(['native:0'])
def test_array_from_xchainer_array(shape, dtype, device):
    t = xchainer.zeros(shape, dtype, 'native:1')
    a = xchainer.array(t)
    assert t is not a
    assert a.shape == shape
    assert a.dtype == dtype
    assert a.device == t.device
    assert a._debug_flat_data == t._debug_flat_data


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_array_from_xchainer_array_with_device(device):
    shape = (2,)
    dtype = xchainer.float64
    t = xchainer.zeros(shape, dtype, 'native:0')
    a = xchainer.array(t, device)
    assert t is not a
    assert a.shape == shape
    assert a.dtype == dtype
    _check_device(a, device)
    assert a._debug_flat_data == t._debug_flat_data


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_empty(xp, shape, dtype, device):
    a = xp.empty(shape, xp.dtype(dtype.name))
    a.fill(0)
    return a


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_empty_with_device(device):
    a = xchainer.empty((2,), 'f', device)
    _check_device(a, device)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_empty_like(xp, shape, dtype, device):
    t = xp.empty(shape, xp.dtype(dtype.name))
    a = xp.empty_like(t)
    a.fill(0)
    return a


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_empty_like_with_device(device):
    t = xchainer.empty((2,), 'f')
    a = xchainer.empty_like(t, device)
    _check_device(a, device)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_zeros(xp, shape, dtype, device):
    return xp.zeros(shape, xp.dtype(dtype.name))


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_zeros_with_device(device):
    a = xchainer.zeros((2,), 'f', device)
    _check_device(a, device)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_zeros_like(xp, shape, dtype, device):
    t = xp.empty(shape, xp.dtype(dtype.name))
    return xp.zeros_like(t)


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_zeros_like_with_device(device):
    t = xchainer.empty((2,), 'f')
    a = xchainer.zeros_like(t, device)
    _check_device(a, device)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_ones(xp, shape, dtype, device):
    return xp.ones(shape, xp.dtype(dtype.name))


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_ones_with_device(device):
    a = xchainer.ones((2,), 'f', device)
    _check_device(a, device)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_ones_like(xp, shape, dtype, device):
    t = xp.empty(shape, xp.dtype(dtype.name))
    return xp.ones_like(t)


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_ones_like_with_device(shape, dtype, device):
    t = xchainer.empty((2,), 'f')
    a = xchainer.ones_like(t, device)
    _check_device(a, device)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('value', [True, False, -2, 0, 1, 2, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_full(xp, shape, value, device):
    return xp.full(shape, value)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('value', [True, False, -2, 0, 1, 2, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_full_with_dtype(xp, shape, dtype, value, device):
    return xp.full(shape, value, xp.dtype(dtype.name))


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
    a = xchainer.full((2,), 1, 'f', device)
    _check_device(a, device)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('value', [True, False, -2, 0, 1, 2, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_full_like(xp, shape, dtype, value, device):
    t = xp.empty(shape, xp.dtype(dtype.name))
    return xp.full_like(t, value)


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_full_like_with_device(device):
    t = xchainer.empty((2,), 'f')
    a = xchainer.full_like(t, 1, device)
    _check_device(a, device)


@pytest.mark.parametrize("stop", [-2, 0, 3, 3.2, False, True])
@pytest.mark.parametrize_device(['native:0'])
def test_arange_stop(xp, stop, dtype, device):
    if dtype.name == 'bool' and stop > 2:  # Checked in test_invalid_arange_too_long_bool
        return xp.array([])
    return xp.arange(stop, dtype=dtype.name)


@pytest.mark.parametrize("start,stop", [
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
def test_arange_start_stop(xp, start, stop, dtype, device):
    if dtype.name == 'bool' and abs(stop - start) > 2:  # Checked in test_invalid_arange_too_long_bool
        return xp.array([])
    return xp.arange(start, stop, dtype=dtype.name)


@pytest.mark.parametrize("start,stop,step", [
    (0, 3, 1),
    (0, 0, 2),
    (0, 1, 2),
    (3, -1, -2),
    (-1, 3, -2),
    (3., 2., 1.2),
    (2., -1., 1.),
    (1, 4, -1.2),
    (4, 1, -1.2),
    (False, True, True),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_arange_start_stop_step(xp, device, start, stop, step, dtype):
    if dtype.name == 'bool' and abs((stop - start) / step) > 2:  # Checked in test_invalid_arange_too_long_bool
        return xp.array([])
    return xp.arange(start, stop, step, dtype=dtype.name)


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_arange_with_device(device):
    _check_device(xchainer.arange(3, device=device), device)
    _check_device(xchainer.arange(3, dtype='f', device=device), device)
    _check_device(xchainer.arange(0, 3, device=device), device)
    _check_device(xchainer.arange(0, 3, dtype='f', device=device), device)
    _check_device(xchainer.arange(0, 3, 2, device=device), device)
    _check_device(xchainer.arange(0, 3, 2, dtype='f', device=device), device)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_invalid_arange_too_long_bool(device):
    def check(xp, err):
        with pytest.raises(err):
            xp.arange(3, dtype='bool')
        with pytest.raises(err):
            xp.arange(1, 4, 1, dtype='bool')
        # Should not raise since the size is <= 2.
        xp.arange(1, 4, 2, dtype='bool')

    check(xchainer, xchainer.DtypeError)
    check(numpy, ValueError)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_invalid_arange_zero_step(device):
    def check(xp, err):
        with pytest.raises(err):
            xp.arange(1, 3, 0)

    check(xchainer, xchainer.XchainerError)
    check(numpy, ZeroDivisionError)
