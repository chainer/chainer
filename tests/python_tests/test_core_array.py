import functools
import itertools
import operator

import pytest

import xchainer


def create_dummy_data(shape, dtype, pattern=1):
    size = functools.reduce(operator.mul, shape, 1)
    if pattern == 1:
        if dtype == xchainer.Dtype.bool:
            return [i % 2 == 1 for i in range(size)]
        else:
            return [i for i in range(size)]
    else:
        if dtype == xchainer.Dtype.bool:
            return [i % 2 == 0 for i in range(size)]
        else:
            return [1 + i for i in range(size)]


@pytest.mark.parametrize('shape,dtype', itertools.product([
    (),
    (0,),
    (1,),
    (1, 1, 1),
    (2, 3),
    (2, 0, 3),
], [
    xchainer.Dtype.bool,
    xchainer.Dtype.int8,
    xchainer.Dtype.int16,
    xchainer.Dtype.int32,
    xchainer.Dtype.int64,
    xchainer.Dtype.uint8,
    xchainer.Dtype.float32,
    xchainer.Dtype.float64,
]))
def test_array(shape, dtype):
    data_list = create_dummy_data(shape, dtype)
    array = xchainer.Array(shape, dtype, data_list)

    assert isinstance(array.shape, xchainer.Shape)
    assert array.shape == shape
    assert array.dtype == dtype
    assert array.offset == 0
    assert array.is_contiguous

    expected_total_size = functools.reduce(operator.mul, shape, 1)
    assert array.total_size == expected_total_size
    assert array.element_bytes == dtype.itemsize
    assert array.total_bytes == dtype.itemsize * expected_total_size

    assert array.debug_flat_data == data_list


def test_array_init_invalid_length():
    with pytest.raises(xchainer.DimensionError):
        xchainer.Array((), xchainer.Dtype.int8, [])

    with pytest.raises(xchainer.DimensionError):
        xchainer.Array((), xchainer.Dtype.int8, [1, 1])

    with pytest.raises(xchainer.DimensionError):
        xchainer.Array((1,), xchainer.Dtype.int8, [])

    with pytest.raises(xchainer.DimensionError):
        xchainer.Array((1,), xchainer.Dtype.int8, [1, 1])

    with pytest.raises(xchainer.DimensionError):
        xchainer.Array((0,), xchainer.Dtype.int8, [1])

    with pytest.raises(xchainer.DimensionError):
        xchainer.Array((3, 2), xchainer.Dtype.int8, [1, 1, 1, 1, 1])

    with pytest.raises(xchainer.DimensionError):
        xchainer.Array((3, 2), xchainer.Dtype.int8, [1, 1, 1, 1, 1, 1, 1])


@pytest.mark.parametrize('shape,dtype', itertools.product([
    (),
    (0,),
    (1,),
    (1, 1, 1),
    (2, 3),
    (2, 0, 3),
], [
    xchainer.Dtype.bool,
    xchainer.Dtype.int8,
    xchainer.Dtype.int16,
    xchainer.Dtype.int32,
    xchainer.Dtype.int64,
    xchainer.Dtype.uint8,
    xchainer.Dtype.float32,
    xchainer.Dtype.float64,
]))
def test_add_iadd(shape, dtype):
    lhs_data_list = create_dummy_data(shape, dtype, pattern=1)
    rhs_data_list = create_dummy_data(shape, dtype, pattern=2)
    lhs = xchainer.Array(shape, dtype, lhs_data_list)
    rhs = xchainer.Array(shape, dtype, rhs_data_list)

    expected_data_list = [x + y for x, y in zip(lhs_data_list, rhs_data_list)]
    if dtype == xchainer.Dtype.bool:
        expected_data_list = [x > 0 for x in expected_data_list]  # [0, 2] => [False, True]

    out = lhs + rhs
    assert out.debug_flat_data == expected_data_list
    assert lhs.debug_flat_data == lhs_data_list
    assert rhs.debug_flat_data == rhs_data_list
    lhs += rhs
    assert lhs.debug_flat_data == expected_data_list
    assert rhs.debug_flat_data == rhs_data_list


@pytest.mark.parametrize('shape,dtype', itertools.product([
    (),
    (0,),
    (1,),
    (1, 1, 1),
    (2, 3),
    (2, 0, 3),
], [
    xchainer.Dtype.bool,
    xchainer.Dtype.int8,
    xchainer.Dtype.int16,
    xchainer.Dtype.int32,
    xchainer.Dtype.int64,
    xchainer.Dtype.uint8,
    xchainer.Dtype.float32,
    xchainer.Dtype.float64,
]))
def test_mul_imul(shape, dtype):
    lhs_data_list = create_dummy_data(shape, dtype, pattern=1)
    rhs_data_list = create_dummy_data(shape, dtype, pattern=2)
    lhs = xchainer.Array(shape, dtype, lhs_data_list)
    rhs = xchainer.Array(shape, dtype, rhs_data_list)

    expected_data_list = [x * y for x, y in zip(lhs_data_list, rhs_data_list)]
    if dtype == xchainer.Dtype.bool:
        expected_data_list = [x > 0 for x in expected_data_list]  # [0, 1] => [False, True]

    out = lhs * rhs
    assert out.debug_flat_data == expected_data_list
    assert lhs.debug_flat_data == lhs_data_list
    assert rhs.debug_flat_data == rhs_data_list
    lhs *= rhs
    assert lhs.debug_flat_data == expected_data_list
    assert rhs.debug_flat_data == rhs_data_list
