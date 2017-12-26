import functools
import operator

import pytest

import xchainer


@pytest.fixture
def inputs(request, shape_data):
    return shape_data['tuple']


def test_attr(inputs):
    tup = inputs
    shape = xchainer.Shape(tup)

    assert shape.ndim == len(tup)
    assert shape.size == len(tup)
    assert str(shape) == str(tup)

    expected_total_size = functools.reduce(operator.mul, tup, 1)
    assert shape.total_size == expected_total_size


def test_eq(inputs):
    tup = inputs
    shape = xchainer.Shape(tup)

    # equality
    assert shape == xchainer.Shape(tup)
    assert shape == tup
    assert tup == shape

    # inequality
    assert shape != xchainer.Shape(tup + (1,))
    assert shape != tup + (1,)
    assert tup + (1,) != shape
    if tup != ():
        assert shape != tuple(['a' for _ in tup])
        # Note: this behavior is different from NumPy
        assert shape != tuple([float(d) for d in tup])
