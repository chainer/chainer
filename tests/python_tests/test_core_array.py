import functools
import operator

import numpy
import pytest

import xchainer


dtypes = [
    xchainer.Dtype.bool,
    xchainer.Dtype.int8,
    xchainer.Dtype.int16,
    xchainer.Dtype.int32,
    xchainer.Dtype.int64,
    xchainer.Dtype.uint8,
    xchainer.Dtype.float32,
    xchainer.Dtype.float64
]


numpy_dtypes = [
    numpy.bool,
    numpy.int8,
    numpy.int16,
    numpy.int32,
    numpy.int64,
    numpy.uint8,
    numpy.float32,
    numpy.float64
]


@pytest.fixture(params=dtypes)
def dtype(request):
    return request.param


@pytest.fixture(params=zip(dtypes, numpy_dtypes))
def dtype_pair(request):
    return request.param


@pytest.fixture(params=[
    (),
    (0,),
    (1,),
    (1, 1, 1),
    (2, 3),
    (2, 0, 3)
])
def shape(request):
    return request.param


@pytest.fixture(params=[
    (xchainer.Dtype.bool, numpy.bool),
    (xchainer.Dtype.int8, numpy.int8),
    (xchainer.Dtype.int16, numpy.int16),
    (xchainer.Dtype.int32, numpy.int32),
    (xchainer.Dtype.int64, numpy.int64),
    (xchainer.Dtype.uint8, numpy.uint8),
    (xchainer.Dtype.float32, numpy.float32),
    (xchainer.Dtype.float64, numpy.float64)
])
def dtypes(request):
    return request.param


def create_dummy_data(shape, dtype, pattern=1):
    size = functools.reduce(operator.mul, shape, 1)
    if pattern == 1:
        if dtype == xchainer.Dtype.bool:
            return [i % 2 == 1 for i in range(size)]
        else:
            return [i for i in range(size)]
    else:
        if dtype == xchainer.Dtype.bool:
            return [i % 3 == 0 for i in range(size)]
        else:
            return [1 + i for i in range(size)]


def create_dummy_ndarray(shape, dtype):
    size = functools.reduce(operator.mul, shape, 1)
    return numpy.arange(size).reshape(shape).astype(dtype)


def test_array_init(shape, dtype):
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


def test_numpy_init(shape, dtype_pair):
    dtype, numpy_dtype = dtype_pair

    data = create_dummy_ndarray(shape, numpy_dtype)
    array = xchainer.Array(data)

    assert isinstance(array.shape, xchainer.Shape)
    assert array.shape == data.shape
    assert array.dtype == dtype
    assert array.offset == 0
    assert array.is_contiguous == data.flags['C_CONTIGUOUS']

    assert array.total_size == data.size
    assert array.element_bytes == data.itemsize
    assert array.total_bytes == data.itemsize * data.size

    assert array.debug_flat_data == data.ravel().tolist()

    # inplace modification
    if data.size > 0:
        data *= create_dummy_ndarray(shape, numpy_dtype)
        assert array.debug_flat_data == data.ravel().tolist()

    # test possibly freed memoty
    data_copy = data.copy()
    del data
    assert array.debug_flat_data == data_copy.ravel().tolist()


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
