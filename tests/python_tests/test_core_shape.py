import functools
import operator

import pytest

import xchainer


@pytest.mark.parametrize('shape_tup', [
    (),
    (0,),
    (1,),
    (1, 1, 1),
    (2, 3),
    (2, 0, 3),
])
def test_shape(shape_tup):
    shape = xchainer.Shape(shape_tup)

    assert shape == xchainer.Shape(shape_tup)
    assert shape == shape_tup
    assert shape_tup == shape
    assert shape.ndim == len(shape_tup)
    assert shape.size == len(shape_tup)

    expected_total_size = functools.reduce(operator.mul, shape_tup, 1)
    assert shape.total_size == expected_total_size
