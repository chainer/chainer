import math

import pytest

import xchainer


def _check_cast_scalar_equals_data(scalar, data):
    assert bool(scalar) == bool(data)
    assert int(scalar) == int(data)
    assert float(scalar) == float(data)


_scalars_data = [
    {'data': -2},
    {'data': 1},
    {'data': -1.5},
    {'data': 2.3},
    {'data': True},
    {'data': False},
]


@pytest.fixture(params=_scalars_data)
def scalar_data(request):
    return request.param


@pytest.fixture
def scalar_init_inputs(request, scalar_data):
    return scalar_data['data']


@pytest.mark.parametrize('value,dtype', [
    (0, xchainer.int64),
    (-1, xchainer.int64),
    (0x7fffffffffffffff, xchainer.int64),
    (-0x8000000000000000, xchainer.int64),
    (0.0, xchainer.float64),
    (float('inf'), xchainer.float64),
    (float('nan'), xchainer.float64),
    (True, xchainer.bool),
    (False, xchainer.bool),
])
def test_init(value, dtype):
    scalar = xchainer.Scalar(value)
    assert scalar.dtype == dtype
    if math.isnan(value):
        assert math.isnan(scalar.tolist())
    else:
        assert scalar.tolist() == value
    assert isinstance(scalar.tolist(), type(value))


@pytest.mark.parametrize('value,cast_dtype,expected_value', [
    (0, xchainer.bool, False),
    (0, xchainer.int8, 0),
    (0, xchainer.int16, 0),
    (0, xchainer.int32, 0),
    (0, xchainer.int64, 0),
    (0, xchainer.uint8, 0),
    (0, xchainer.float32, 0.0),
    (0, xchainer.float64, 0.0),
    (0.0, xchainer.bool, False),
    (0.0, xchainer.int8, 0),
    (0.0, xchainer.int16, 0),
    (0.0, xchainer.int32, 0),
    (0.0, xchainer.int64, 0),
    (0.0, xchainer.uint8, 0),
    (0.0, xchainer.float32, 0.0),
    (0.0, xchainer.float64, 0.0),
    (1, xchainer.bool, True),
    (1, xchainer.int8, 1),
    (1, xchainer.int16, 1),
    (1, xchainer.int32, 1),
    (1, xchainer.int64, 1),
    (1, xchainer.uint8, 1),
    (1, xchainer.float32, 1.0),
    (1, xchainer.float64, 1.0),
    (1.0, xchainer.bool, True),
    (1.0, xchainer.int8, 1),
    (1.0, xchainer.int16, 1),
    (1.0, xchainer.int32, 1),
    (1.0, xchainer.int64, 1),
    (1.0, xchainer.uint8, 1),
    (1.0, xchainer.float32, 1.0),
    (1.0, xchainer.float64, 1.0),
    (-1, xchainer.bool, True),
    (-1, xchainer.int8, -1),
    (-1, xchainer.int16, -1),
    (-1, xchainer.int32, -1),
    (-1, xchainer.int64, -1),
    (-1, xchainer.uint8, 0xff),
    (-1, xchainer.float32, -1.0),
    (-1, xchainer.float64, -1.0),
    (0x100, xchainer.bool, True),
    (0x100, xchainer.int8, 0),
    (0x100, xchainer.int16, 0x100),
    (0x100, xchainer.int32, 0x100),
    (0x100, xchainer.int64, 0x100),
    (0x100, xchainer.uint8, 0),
    (0x10000, xchainer.bool, True),
    (0x10000, xchainer.int8, 0),
    (0x10000, xchainer.int16, 0),
    (0x10000, xchainer.int32, 0x10000),
    (0x10000, xchainer.int64, 0x10000),
    (0x10000, xchainer.uint8, 0),
    (0x100000000, xchainer.bool, True),
    (0x100000000, xchainer.int8, 0),
    (0x100000000, xchainer.int16, 0),
    (0x100000000, xchainer.int32, 0),
    (0x100000000, xchainer.int64, 0x100000000),
    (0x100000000, xchainer.uint8, 0),
    (0x7fffffffffffffff, xchainer.bool, True),
    (0x7fffffffffffffff, xchainer.int8, -1),
    (0x7fffffffffffffff, xchainer.int16, -1),
    (0x7fffffffffffffff, xchainer.int32, -1),
    (0x7fffffffffffffff, xchainer.int64, 0x7fffffffffffffff),
    (0x7fffffffffffffff, xchainer.uint8, 255),
])
def test_init_casted(value, cast_dtype, expected_value):
    scalar = xchainer.Scalar(value, cast_dtype)
    assert scalar.dtype == cast_dtype
    if math.isnan(expected_value):
        assert math.isnan(scalar.tolist())
    else:
        assert scalar.tolist() == expected_value
    assert isinstance(scalar.tolist(), type(expected_value))


def test_cast(scalar_init_inputs):
    data = scalar_init_inputs
    scalar = xchainer.Scalar(data)

    _check_cast_scalar_equals_data(scalar, data)
    _check_cast_scalar_equals_data(+scalar, +data)
    if isinstance(data, bool):
        with pytest.raises(xchainer.DtypeError):
            -scalar  # should not be able to negate bool
    else:
        _check_cast_scalar_equals_data(-scalar, -data)


def test_dtype(scalar_init_inputs):
    data = scalar_init_inputs
    scalar = xchainer.Scalar(data)

    if isinstance(data, bool):
        assert scalar.dtype == xchainer.bool
    elif isinstance(data, int):
        assert scalar.dtype == xchainer.int64
    elif isinstance(data, float):
        assert scalar.dtype == xchainer.float64
    else:
        assert False


def test_repr(scalar_init_inputs):
    data = scalar_init_inputs
    scalar = xchainer.Scalar(data)

    assert repr(scalar) == repr(data)
    assert str(scalar) == str(data)


def test_init_invalid():
    with pytest.raises(TypeError):
        xchainer.Scalar("1")  # string, which is not a numeric
