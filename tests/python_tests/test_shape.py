import functools
import operator

import pytest

import xchainer


_shapes_data = [
    {'tuple': ()},
    {'tuple': (0,)},
    {'tuple': (1,)},
    {'tuple': (2, 3)},
    {'tuple': (1, 1, 1)},
    {'tuple': (2, 0, 3)},
]


@pytest.fixture(params=_shapes_data)
def shape_data(request):
    return request.param


@pytest.fixture
def shape_init_inputs(request, shape_data):
    return shape_data['tuple']


def test_attr(shape_init_inputs):
    tup = shape_init_inputs
    shape = xchainer.Shape(tup)

    assert shape.ndim == len(tup)
    assert shape.size == len(tup)
    assert str(shape) == str(tup)

    expected_total_size = functools.reduce(operator.mul, tup, 1)
    assert shape.total_size == expected_total_size


def test_eq(shape_init_inputs):
    tup = shape_init_inputs
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
