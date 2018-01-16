import math

import pytest

import xchainer


def _check_cast_scalar_equals_data(scalar, data):
    assert bool(scalar) == bool(data)
    assert int(scalar) == int(data)
    assert float(scalar) == float(data)


all_scalar_values = [
    -2, 1, -1.5, 2.3, True, False, float('inf'), float('nan')]


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


@pytest.mark.parametrize('value1,value2', [
    # TODO(niboshi): Support commented-out cases
    (0, 0),
    (1, 1),
    # (1, 1.0),
    (1.5, 1.5),
    (-1.5, -1.5),
    (True, True),
    (False, False),
    # (True, 1),
    # (True, 1.0),
    # (False, 0),
    # (False, 0.0),
    # (float('inf'), float('inf')),
])
def test_equality(value1, value2):
    scalar1 = xchainer.Scalar(value1)
    scalar2 = xchainer.Scalar(value2)

    assert scalar1 == scalar2
    assert scalar2 == scalar1

    assert scalar1 == value1
    assert value1 == scalar1

    assert scalar2 == value2
    assert value2 == scalar2

    assert scalar2 == value1
    assert value1 == scalar2

    assert scalar1 == value2
    assert value2 == scalar1


@pytest.mark.parametrize('value1,value2', [
    (0, 1),
    (-1, 1),
    (-1.0001, -1.0),
    (-1.0001, -1),
    (True, False),
    (True, 1.1),
    (1.0001, 1.0002),
    (float('nan'), float('nan')),
])
def test_inequality(value1, value2):
    scalar1 = xchainer.Scalar(value1)
    scalar2 = xchainer.Scalar(value2)

    assert scalar1 != scalar2
    assert scalar2 != scalar1

    assert scalar2 != value1
    assert value1 != scalar2

    assert scalar1 != value2
    assert value2 != scalar1


@pytest.mark.parametrize('value', [
    -2, 1, -1.5, 2.3, True, False
])
def test_cast(value):
    scalar = xchainer.Scalar(value)

    _check_cast_scalar_equals_data(scalar, value)
    _check_cast_scalar_equals_data(+scalar, +value)
    if isinstance(value, bool):
        with pytest.raises(xchainer.DtypeError):
            -scalar  # should not be able to negate bool
    else:
        _check_cast_scalar_equals_data(-scalar, -value)


@pytest.mark.parametrize('value', all_scalar_values)
def test_dtype(value):
    scalar = xchainer.Scalar(value)

    if isinstance(value, bool):
        assert scalar.dtype == xchainer.bool
    elif isinstance(value, int):
        assert scalar.dtype == xchainer.int64
    elif isinstance(value, float):
        assert scalar.dtype == xchainer.float64
    else:
        assert False


@pytest.mark.parametrize('value', all_scalar_values)
def test_repr(value):
    scalar = xchainer.Scalar(value)

    assert repr(scalar) == repr(value)
    assert str(scalar) == str(value)


def test_init_invalid():
    with pytest.raises(TypeError):
        xchainer.Scalar("1")  # string, which is not a numeric
