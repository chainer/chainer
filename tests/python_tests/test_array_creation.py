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


def check_basic_creation(a, shape, dtype):
    assert a.shape == shape
    assert a.dtype == dtype
    assert a.is_contiguous
    assert a.offset == 0
    assert a.total_size == functools.reduce(operator.mul, shape, 1)
    assert not a.is_grad_required()


def test_empty(shape, dtype):
    a = xchainer.empty(shape, dtype)
    check_basic_creation(a, shape, dtype)


def test_empty_cuda(shape, dtype):
    device = xchainer.get_default_context().get_device('cuda', 0)
    with xchainer.device_scope(device):
        a = xchainer.empty(shape, dtype)
    check_basic_creation(a, shape, dtype)


def test_empty_like(shape, dtype):
    t = create_dummy_array(shape, dtype)
    a = xchainer.empty_like(t)
    check_basic_creation(a, shape, dtype)

    assert a._debug_data_memory_address != t._debug_data_memory_address, 'memory must not be shared'


def test_zeros(shape, dtype):
    a = xchainer.zeros(shape, dtype)
    check_basic_creation(a, shape, dtype)

    value = False if dtype == xchainer.bool else 0
    assert all([el == value for el in a._debug_flat_data])


def test_zeros_like(shape, dtype):
    t = create_dummy_array(shape, dtype)
    a = xchainer.zeros_like(t)
    check_basic_creation(a, shape, dtype)

    value = False if dtype == xchainer.bool else 0
    assert all([el == value for el in a._debug_flat_data])

    assert a._debug_data_memory_address != t._debug_data_memory_address, 'memory must not be shared'


def test_ones(shape, dtype):
    a = xchainer.ones(shape, dtype)
    check_basic_creation(a, shape, dtype)

    value = True if dtype == xchainer.bool else 1
    assert all([el == value for el in a._debug_flat_data])


def test_ones_like(shape, dtype):
    t = create_dummy_array(shape, dtype)
    a = xchainer.ones_like(t)
    check_basic_creation(a, shape, dtype)

    value = True if dtype == xchainer.bool else 1
    assert all([el == value for el in a._debug_flat_data])

    assert a._debug_data_memory_address != t._debug_data_memory_address, 'memory must not be shared'


def check_full(shape, value, dtype):
    a = xchainer.full(shape, value, dtype)
    check_basic_creation(a, shape, dtype)

    if math.isnan(value):
        assert all([math.isnan(el) for el in a._debug_flat_data])
    else:
        assert a._debug_flat_data == [value] * a.total_size


def check_full_with_scalar(shape, scalar):
    assert isinstance(scalar, xchainer.Scalar), 'This test must be done on xchainer.Scalar'

    a = xchainer.full(shape, scalar)
    check_basic_creation(a, shape, scalar.dtype)

    if scalar.dtype in (xchainer.float32, xchainer.float64) and math.isnan(float(scalar)):
        assert all([math.isnan(el) for el in a._debug_flat_data])
    else:
        assert a._debug_flat_data == [scalar.tolist()] * a.total_size


def check_full_with_py_scalar(shape, value):
    if isinstance(value, bool):
        dtype = xchainer.bool_
    elif isinstance(value, int):
        dtype = xchainer.int64
    elif isinstance(value, float):
        dtype = xchainer.float64
    else:
        assert False, 'This test should be done on either of (bool, int, float)'

    a = xchainer.full(shape, value)
    check_basic_creation(a, shape, dtype)

    if isinstance(value, float) and math.isnan(value):
        assert all([math.isnan(el) for el in a._debug_flat_data])
    else:
        assert a._debug_flat_data == [value] * a.total_size


def check_full_like(shape, value, dtype):
    t = create_dummy_array(shape, dtype)
    a = xchainer.full_like(t, value)
    check_basic_creation(a, shape, dtype)

    if math.isnan(value):
        assert all([math.isnan(el) for el in a._debug_flat_data])
    else:
        assert a._debug_flat_data == [value] * a.total_size

    assert a._debug_data_memory_address != t._debug_data_memory_address, 'memory must not be shared'


def test_full_full_like_0(shape, dtype):
    value = False if dtype == xchainer.bool else 0
    check_full(shape, value, dtype)
    check_full_with_scalar(shape, xchainer.Scalar(value, dtype))
    check_full_like(shape, value, dtype)


def test_full_full_like_1(shape, dtype):
    value = True if dtype == xchainer.bool else 1
    check_full(shape, value, dtype)
    check_full_with_scalar(shape, xchainer.Scalar(value, dtype))
    check_full_like(shape, value, dtype)


def test_full_full_like_neg1(shape, signed_dtype):
    dtype = signed_dtype
    value = -1
    check_full(shape, value, dtype)
    check_full_with_scalar(shape, xchainer.Scalar(value, dtype))
    check_full_like(shape, value, dtype)


def test_full_full_like_nan(shape, float_dtype):
    dtype = float_dtype
    value = float('nan')
    check_full(shape, value, dtype)
    check_full_with_scalar(shape, xchainer.Scalar(value, dtype))
    check_full_like(shape, value, dtype)


def test_full_full_like_inf(shape, float_dtype):
    dtype = float_dtype
    value = float('inf')
    check_full(shape, value, dtype)
    check_full_with_scalar(shape, xchainer.Scalar(value, dtype))
    check_full_like(shape, value, dtype)
