import functools
import math
import operator

import pytest

import xchainer


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


def create_dummy_array(shape, dtype):
    size = functools.reduce(operator.mul, shape, 1)
    if dtype == xchainer.bool:
        data_list = [i % 2 == 1 for i in range(size)]
    else:
        data_list = [i for i in range(size)]
    return xchainer.Array(shape, dtype, data_list)


def check_basic_creation(a, shape, dtype, device_id=None):
    assert a.shape == shape
    assert a.dtype == dtype
    assert a.is_contiguous
    assert a.offset == 0
    assert a.total_size == functools.reduce(operator.mul, shape, 1)
    assert not a.is_grad_required()
    if device_id is None:
        device = xchainer.get_default_device()
    else:
        device = xchainer.get_default_context().get_device(device_id)
    assert a.device is device


def test_empty(shape, dtype):
    def _check(a):
        check_basic_creation(a, shape, dtype)

    _check(xchainer.empty(shape, dtype))
    _check(xchainer.empty(shape=shape, dtype=dtype))


def test_empty_device(shape, dtype):
    def _check(a):
        check_basic_creation(a, shape, dtype, 'native:1')

    _check(xchainer.empty(shape, dtype, 'native:1'))
    _check(xchainer.empty(shape=shape, dtype=dtype, device='native:1'))


def test_empty_like(shape, dtype):
    t = create_dummy_array(shape, dtype)

    def _check(a):
        check_basic_creation(a, shape, dtype)
        assert a._debug_data_memory_address != t._debug_data_memory_address, 'memory must not be shared'

    _check(xchainer.empty_like(t))
    _check(xchainer.empty_like(a=t))


def test_empty_like_device(shape, dtype):
    t = create_dummy_array(shape, dtype)

    def _check(a):
        check_basic_creation(a, shape, dtype, 'native:1')
        assert a._debug_data_memory_address != t._debug_data_memory_address, 'memory must not be shared'

    _check(xchainer.empty_like(t, 'native:1'))
    _check(xchainer.empty_like(a=t, device='native:1'))


def test_zeros(shape, dtype):
    def _check(a):
        check_basic_creation(a, shape, dtype)

        value = False if dtype == xchainer.bool else 0
        assert all([el == value for el in a._debug_flat_data])

    _check(xchainer.zeros(shape, dtype))
    _check(xchainer.zeros(shape=shape, dtype=dtype))


def test_zeros_device(shape, dtype):
    def _check(a):
        check_basic_creation(a, shape, dtype, 'native:1')

        value = False if dtype == xchainer.bool else 0
        assert all([el == value for el in a._debug_flat_data])

    _check(xchainer.zeros(shape, dtype, 'native:1'))
    _check(xchainer.zeros(shape=shape, dtype=dtype, device='native:1'))


def test_zeros_like(shape, dtype):
    t = create_dummy_array(shape, dtype)

    def _check(a):
        check_basic_creation(a, shape, dtype)

        value = False if dtype == xchainer.bool else 0
        assert all([el == value for el in a._debug_flat_data])

        assert a._debug_data_memory_address != t._debug_data_memory_address, 'memory must not be shared'

    _check(xchainer.zeros_like(t))
    _check(xchainer.zeros_like(a=t))


def test_zeros_like_device(shape, dtype):
    t = create_dummy_array(shape, dtype)

    def _check(a):
        check_basic_creation(a, shape, dtype, 'native:1')

        value = False if dtype == xchainer.bool else 0
        assert all([el == value for el in a._debug_flat_data])

        assert a._debug_data_memory_address != t._debug_data_memory_address, 'memory must not be shared'

    _check(xchainer.zeros_like(t, 'native:1'))
    _check(xchainer.zeros_like(a=t, device='native:1'))


def test_ones(shape, dtype):
    def _check(a):
        check_basic_creation(a, shape, dtype)

        value = True if dtype == xchainer.bool else 1
        assert all([el == value for el in a._debug_flat_data])

    _check(xchainer.ones(shape, dtype))
    _check(xchainer.ones(shape=shape, dtype=dtype))


def test_ones_device(shape, dtype):
    def _check(a):
        check_basic_creation(a, shape, dtype, 'native:1')

        value = True if dtype == xchainer.bool else 1
        assert all([el == value for el in a._debug_flat_data])

    _check(xchainer.ones(shape, dtype, 'native:1'))
    _check(xchainer.ones(shape=shape, dtype=dtype, device='native:1'))


def test_ones_like(shape, dtype):
    t = create_dummy_array(shape, dtype)

    def _check(a):
        check_basic_creation(a, shape, dtype)

        value = True if dtype == xchainer.bool else 1
        assert all([el == value for el in a._debug_flat_data])

        assert a._debug_data_memory_address != t._debug_data_memory_address, 'memory must not be shared'

    _check(xchainer.ones_like(t))
    _check(xchainer.ones_like(a=t))


def test_ones_like_device(shape, dtype):
    t = create_dummy_array(shape, dtype)

    def _check(a):
        check_basic_creation(a, shape, dtype, 'native:1')

        value = True if dtype == xchainer.bool else 1
        assert all([el == value for el in a._debug_flat_data])

        assert a._debug_data_memory_address != t._debug_data_memory_address, 'memory must not be shared'

    _check(xchainer.ones_like(t, 'native:1'))
    _check(xchainer.ones_like(a=t, device='native:1'))


def check_full(shape, value, dtype, device=None):
    def _check(a):
        check_basic_creation(a, shape, dtype, device)

        if math.isnan(value):
            assert all([math.isnan(el) for el in a._debug_flat_data])
        else:
            assert a._debug_flat_data == [value] * a.total_size

    if device is None:
        _check(xchainer.full(shape, value, dtype))
        _check(xchainer.full(shape=shape, fill_value=value, dtype=dtype))
    else:
        _check(xchainer.full(shape, value, dtype, device))
        _check(xchainer.full(shape=shape, fill_value=value, dtype=dtype, device=device))


def check_full_with_scalar(shape, scalar, device=None):
    assert isinstance(scalar, xchainer.Scalar), 'This test must be done on xchainer.Scalar'

    if device is None:
        a = xchainer.full(shape, scalar)
    else:
        a = xchainer.full(shape, scalar, device)

    check_basic_creation(a, shape, scalar.dtype, device)

    if scalar.dtype in (xchainer.float32, xchainer.float64) and math.isnan(float(scalar)):
        assert all([math.isnan(el) for el in a._debug_flat_data])
    else:
        assert a._debug_flat_data == [scalar.tolist()] * a.total_size


def check_full_with_py_scalar(shape, value, device=None):
    if isinstance(value, bool):
        dtype = xchainer.bool_
    elif isinstance(value, int):
        dtype = xchainer.int64
    elif isinstance(value, float):
        dtype = xchainer.float64
    else:
        assert False, 'This test should be done on either of (bool, int, float)'

    if device is None:
        a = xchainer.full(shape, value)
    else:
        a = xchainer.full(shape, value, device)

    check_basic_creation(a, shape, dtype, device)

    if isinstance(value, float) and math.isnan(value):
        assert all([math.isnan(el) for el in a._debug_flat_data])
    else:
        assert a._debug_flat_data == [value] * a.total_size


def check_full_like(shape, value, dtype, device=None):
    t = create_dummy_array(shape, dtype)

    def _check(a):
        check_basic_creation(a, shape, dtype, device)

        if math.isnan(value):
            assert all([math.isnan(el) for el in a._debug_flat_data])
        else:
            assert a._debug_flat_data == [value] * a.total_size

        assert a._debug_data_memory_address != t._debug_data_memory_address, 'memory must not be shared'

    if device is None:
        _check(xchainer.full_like(t, value))
        _check(xchainer.full_like(a=t, fill_value=value))
    else:
        _check(xchainer.full_like(t, value, device))
        _check(xchainer.full_like(a=t, fill_value=value, device=device))


def test_full_full_like_0(shape, dtype):
    value = False if dtype == xchainer.bool else 0
    check_full(shape, value, dtype)
    check_full(shape, value, dtype, 'native:1')
    check_full_with_scalar(shape, xchainer.Scalar(value, dtype))
    check_full_with_scalar(shape, xchainer.Scalar(value, dtype), 'native:1')
    check_full_like(shape, value, dtype)
    check_full_like(shape, value, dtype, 'native:1')


def test_full_full_like_1(shape, dtype):
    value = True if dtype == xchainer.bool else 1
    check_full(shape, value, dtype)
    check_full(shape, value, dtype, 'native:1')
    check_full_with_scalar(shape, xchainer.Scalar(value, dtype))
    check_full_with_scalar(shape, xchainer.Scalar(value, dtype), 'native:1')
    check_full_like(shape, value, dtype)
    check_full_like(shape, value, dtype, 'native:1')


def test_full_full_like_neg1(shape, signed_dtype):
    dtype = signed_dtype
    value = -1
    check_full(shape, value, dtype)
    check_full(shape, value, dtype, 'native:1')
    check_full_with_scalar(shape, xchainer.Scalar(value, dtype))
    check_full_with_scalar(shape, xchainer.Scalar(value, dtype), 'native:1')
    check_full_like(shape, value, dtype)
    check_full_like(shape, value, dtype, 'native:1')


def test_full_full_like_nan(shape, float_dtype):
    dtype = float_dtype
    value = float('nan')
    check_full(shape, value, dtype)
    check_full(shape, value, dtype, 'native:1')
    check_full_with_scalar(shape, xchainer.Scalar(value, dtype))
    check_full_with_scalar(shape, xchainer.Scalar(value, dtype), 'native:1')
    check_full_like(shape, value, dtype)
    check_full_like(shape, value, dtype, 'native:1')


def test_full_full_like_inf(shape, float_dtype):
    dtype = float_dtype
    value = float('inf')
    check_full(shape, value, dtype)
    check_full(shape, value, dtype, 'native:1')
    check_full_with_scalar(shape, xchainer.Scalar(value, dtype))
    check_full_with_scalar(shape, xchainer.Scalar(value, dtype), 'native:1')
    check_full_like(shape, value, dtype)
    check_full_like(shape, value, dtype, 'native:1')
