import pytest

import xchainer


def _check_cast_scalar_equals_data(scalar, data):
    assert bool(scalar) == bool(data)
    assert int(scalar) == int(data)
    assert float(scalar) == float(data)


all_scalar_values = [
    -2, 1, -1.5, 2.3, True, False, float('inf'), float('nan')]


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
