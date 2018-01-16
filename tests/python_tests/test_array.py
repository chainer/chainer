import functools
import operator

import numpy
import pytest

import xchainer


def _create_dummy_data(shape_tup, dtype, pattern=1):
    size = _size(shape_tup)
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


def _create_dummy_ndarray(shape_tup, numpy_dtype):
    return numpy.arange(_size(shape_tup)).reshape(shape_tup).astype(numpy_dtype)


def _check_array(array, expected_dtype, expected_shape, expected_total_size, expected_data_list):
    assert isinstance(array.dtype, xchainer.Dtype)
    assert isinstance(array.shape, xchainer.Shape)
    assert array.dtype == expected_dtype
    assert array.shape == expected_shape
    assert array.element_bytes == expected_dtype.itemsize
    assert array.total_size == expected_total_size
    assert array.total_bytes == expected_dtype.itemsize * expected_total_size
    assert array.debug_flat_data == expected_data_list
    assert array.is_contiguous
    assert array.offset == 0


def _check_array_equals_ndarray(array, ndarray):
    assert array.shape == ndarray.shape
    assert array.total_size == ndarray.size
    assert array.ndim == ndarray.ndim
    assert array.element_bytes == ndarray.itemsize
    assert array.total_bytes == ndarray.itemsize * ndarray.size
    assert array.debug_flat_data == ndarray.ravel().tolist()
    assert array.is_contiguous == ndarray.flags['C_CONTIGUOUS']


def _check_ndarray_equal_ndarray(ndarray1, ndarray2):
    assert ndarray1.shape == ndarray2.shape
    assert ndarray1.size == ndarray2.size
    assert ndarray1.ndim == ndarray2.ndim
    assert ndarray1.itemsize == ndarray2.itemsize
    assert ndarray1.strides == ndarray2.strides
    assert numpy.array_equal(ndarray1, ndarray2)
    assert ndarray1.dtype == ndarray2.dtype
    assert ndarray1.flags == ndarray2.flags


def _size(tup):
    return functools.reduce(operator.mul, tup, 1)


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
def array_init_inputs(shape_data, dtype):
    shape_tup = shape_data['tuple']
    return shape_tup, dtype


def test_init(array_init_inputs):
    shape_tup, dtype = array_init_inputs

    shape = xchainer.Shape(shape_tup)

    data_list = _create_dummy_data(shape_tup, dtype)

    array = xchainer.Array(shape, dtype, data_list)

    _check_array(array, dtype, shape, _size(shape_tup), data_list)


def test_numpy_init(array_init_inputs):
    shape_tup, dtype = array_init_inputs

    shape = xchainer.Shape(shape_tup)

    numpy_dtype = getattr(numpy, dtype.name)

    ndarray = _create_dummy_ndarray(shape_tup, numpy_dtype)

    array = xchainer.Array(ndarray)

    _check_array(array, dtype, shape, _size(shape_tup), ndarray.ravel().tolist())
    _check_array_equals_ndarray(array, ndarray)

    _check_array_equals_ndarray(array, ndarray)

    # test possibly freed memory
    data_copy = ndarray.copy()
    del ndarray
    assert array.debug_flat_data == data_copy.ravel().tolist()

    # recovered data should be equal
    data_recovered = numpy.array(array)
    _check_ndarray_equal_ndarray(data_copy, data_recovered)

    # recovered data should be a copy
    data_recovered_to_modify = numpy.array(array)
    data_recovered_to_modify *= _create_dummy_ndarray(shape_tup, numpy_dtype)
    _check_array_equals_ndarray(array, data_recovered)


def test_add_iadd(array_init_inputs):
    shape_tup, dtype = array_init_inputs

    shape = xchainer.Shape(shape_tup)

    lhs_data_list = _create_dummy_data(shape_tup, dtype, pattern=1)
    rhs_data_list = _create_dummy_data(shape_tup, dtype, pattern=2)

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


def test_mul_imul(array_init_inputs):
    shape_tup, dtype = array_init_inputs

    shape = xchainer.Shape(shape_tup)

    lhs_data_list = _create_dummy_data(shape_tup, dtype, pattern=1)
    rhs_data_list = _create_dummy_data(shape_tup, dtype, pattern=2)

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


def test_array_repr():
    array = xchainer.Array((0,), xchainer.Dtype.bool, [])
    assert str(array) == 'array([], dtype=bool)'

    array = xchainer.Array((1,), xchainer.Dtype.bool, [False])
    assert str(array) == 'array([False], dtype=bool)'

    array = xchainer.Array((2, 3), xchainer.Dtype.int8, [0, 1, 2, 3, 4, 5])
    assert str(array) == (
        'array([[0, 1, 2],\n'
        '       [3, 4, 5]], dtype=int8)'
    )

    array = xchainer.Array((2, 3), xchainer.Dtype.float32, [0, 1, 2, 3.25, 4, 5])
    assert str(array) == (
        'array([[0.  , 1.  , 2.  ],\n'
        '       [3.25, 4.  , 5.  ]], dtype=float32)'
    )


def test_array_property_requires_grad():
    array = xchainer.Array((3, 1), xchainer.Dtype.int8, [1, 1, 1])
    assert not array.requires_grad
    array.requires_grad = True
    assert array.requires_grad
    array.requires_grad = False
    assert not array.requires_grad


def test_array_grad():
    array = xchainer.Array((3, 1), xchainer.Dtype.int8, [1, 1, 1])
    grad = xchainer.Array((3, 1), xchainer.Dtype.float32, [0.5, 0.5, 0.5])
    assert not array.grad
    array.grad = grad
    assert array.grad.debug_flat_data == grad.debug_flat_data
    array.clear_grad()
    assert not array.grad
