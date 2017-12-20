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

    assert shape.ndim == len(shape_tup)
    assert shape.size == len(shape_tup)

    # equality
    assert shape == xchainer.Shape(shape_tup)
    assert shape == shape_tup
    assert shape_tup == shape
    # inequality
    assert shape != xchainer.Shape(shape_tup + (1,))
    assert shape != shape_tup + (1,)
    assert shape_tup + (1,) != shape_tup
    if shape_tup != ():
        assert shape != tuple(['a' for _ in shape_tup])
        # Note: this behavior is different from NumPy
        assert shape != tuple([float(d) for d in shape_tup])

    expected_total_size = functools.reduce(operator.mul, shape_tup, 1)
    assert shape.total_size == expected_total_size
