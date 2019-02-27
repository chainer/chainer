import math

import pytest

import chainerx


def _check_cast_scalar_equals_data(scalar, data):
    assert bool(scalar) == bool(data)
    assert int(scalar) == int(data)
    assert float(scalar) == float(data)


all_scalar_values = [
    -2, 1, -1.5, 2.3, True, False, float('inf'), float('nan')]


@pytest.mark.parametrize('value,dtype', [
    (0, chainerx.int64),
    (-1, chainerx.int64),
    (0x7fffffffffffffff, chainerx.int64),
    (-0x8000000000000000, chainerx.int64),
    (0.0, chainerx.float64),
    (float('inf'), chainerx.float64),
    (float('nan'), chainerx.float64),
    (True, chainerx.bool_),
    (False, chainerx.bool_),
])
def test_init_without_dtype(value, dtype):
    scalar = chainerx.Scalar(value)
    assert scalar.dtype == dtype
    if math.isnan(value):
        assert math.isnan(scalar.tolist())
    else:
        assert scalar.tolist() == value
    assert isinstance(scalar.tolist(), type(value))


@pytest.mark.parametrize('value,cast_dtype,expected_value', [
    (0, chainerx.bool_, False),
    (0, chainerx.int8, 0),
    (0, chainerx.int16, 0),
    (0, chainerx.int32, 0),
    (0, chainerx.int64, 0),
    (0, chainerx.uint8, 0),
    (0, chainerx.float32, 0.0),
    (0, chainerx.float64, 0.0),
    (0.0, chainerx.bool_, False),
    (0.0, chainerx.int8, 0),
    (0.0, chainerx.int16, 0),
    (0.0, chainerx.int32, 0),
    (0.0, chainerx.int64, 0),
    (0.0, chainerx.uint8, 0),
    (0.0, chainerx.float32, 0.0),
    (0.0, chainerx.float64, 0.0),
    (1, chainerx.bool_, True),
    (1, chainerx.int8, 1),
    (1, chainerx.int16, 1),
    (1, chainerx.int32, 1),
    (1, chainerx.int64, 1),
    (1, chainerx.uint8, 1),
    (1, chainerx.float32, 1.0),
    (1, chainerx.float64, 1.0),
    (1.0, chainerx.bool_, True),
    (1.0, chainerx.int8, 1),
    (1.0, chainerx.int16, 1),
    (1.0, chainerx.int32, 1),
    (1.0, chainerx.int64, 1),
    (1.0, chainerx.uint8, 1),
    (1.0, chainerx.float32, 1.0),
    (1.0, chainerx.float64, 1.0),
    (-1, chainerx.bool_, True),
    (-1, chainerx.int8, -1),
    (-1, chainerx.int16, -1),
    (-1, chainerx.int32, -1),
    (-1, chainerx.int64, -1),
    (-1, chainerx.uint8, 0xff),
    (-1, chainerx.float32, -1.0),
    (-1, chainerx.float64, -1.0),
    (0x100, chainerx.bool_, True),
    (0x100, chainerx.int8, 0),
    (0x100, chainerx.int16, 0x100),
    (0x100, chainerx.int32, 0x100),
    (0x100, chainerx.int64, 0x100),
    (0x100, chainerx.uint8, 0),
    (0x10000, chainerx.bool_, True),
    (0x10000, chainerx.int8, 0),
    (0x10000, chainerx.int16, 0),
    (0x10000, chainerx.int32, 0x10000),
    (0x10000, chainerx.int64, 0x10000),
    (0x10000, chainerx.uint8, 0),
    (0x100000000, chainerx.bool_, True),
    (0x100000000, chainerx.int8, 0),
    (0x100000000, chainerx.int16, 0),
    (0x100000000, chainerx.int32, 0),
    (0x100000000, chainerx.int64, 0x100000000),
    (0x100000000, chainerx.uint8, 0),
    (0x7fffffffffffffff, chainerx.bool_, True),
    (0x7fffffffffffffff, chainerx.int8, -1),
    (0x7fffffffffffffff, chainerx.int16, -1),
    (0x7fffffffffffffff, chainerx.int32, -1),
    (0x7fffffffffffffff, chainerx.int64, 0x7fffffffffffffff),
    (0x7fffffffffffffff, chainerx.uint8, 255),
])
def test_init_casted(value, cast_dtype, expected_value):
    scalar = chainerx.Scalar(value, cast_dtype)
    assert scalar.dtype == cast_dtype
    if math.isnan(expected_value):
        assert math.isnan(scalar.tolist())
    else:
        assert scalar.tolist() == expected_value
    assert isinstance(scalar.tolist(), type(expected_value))


@pytest.mark.parametrize(
    'value',
    [0, 0.0, 1, 1.0, -1, 0x100, 0x10000, 0x100000000, 0x7fffffffffffffff])
@chainerx.testing.parametrize_dtype_specifier('dtype_spec')
def test_init_with_dtype(value, dtype_spec):
    expected_dtype = chainerx.dtype(dtype_spec)
    scalar = chainerx.Scalar(value, dtype_spec)
    assert scalar.dtype == expected_dtype
    assert scalar == chainerx.Scalar(value, expected_dtype)


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
    scalar1 = chainerx.Scalar(value1)
    scalar2 = chainerx.Scalar(value2)

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
    scalar1 = chainerx.Scalar(value1)
    scalar2 = chainerx.Scalar(value2)

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
    scalar = chainerx.Scalar(value)

    _check_cast_scalar_equals_data(scalar, value)
    _check_cast_scalar_equals_data(+scalar, +value)
    if isinstance(value, bool):
        with pytest.raises(chainerx.DtypeError):
            -scalar  # should not be able to negate bool
    else:
        _check_cast_scalar_equals_data(-scalar, -value)


@pytest.mark.parametrize('value', all_scalar_values)
def test_dtype(value):
    scalar = chainerx.Scalar(value)

    if isinstance(value, bool):
        assert scalar.dtype == chainerx.bool_
    elif isinstance(value, int):
        assert scalar.dtype == chainerx.int64
    elif isinstance(value, float):
        assert scalar.dtype == chainerx.float64
    else:
        assert False


@pytest.mark.parametrize('value', all_scalar_values)
def test_repr(value):
    scalar = chainerx.Scalar(value)

    assert repr(scalar) == repr(value)
    assert str(scalar) == str(value)


def test_init_invalid():
    with pytest.raises(TypeError):
        chainerx.Scalar('1')  # string, which is not a numeric
