import functools
import math
import operator

import numpy
import pytest

import xchainer
import xchainer.testing


_shapes = [
    (),
    (0,),
    (1,),
    (2, 3),
    (1, 1, 1),
    (2, 0, 3),
]


@pytest.fixture(params=_shapes)
def shape(request):
    return request.param


def _create_dummy_data(shape, dtype, pattern=1):
    assert isinstance(dtype, str)

    size = _size(shape)
    if pattern == 1:
        if dtype in ('bool', 'bool_'):
            return [i % 2 == 1 for i in range(size)]
        else:
            return [i for i in range(size)]
    else:
        if dtype in ('bool', 'bool_'):
            return [i % 3 == 0 for i in range(size)]
        else:
            return [1 + i for i in range(size)]


def _create_dummy_ndarray(shape, dtype):
    assert isinstance(dtype, str)
    return numpy.arange(_size(shape)).reshape(shape).astype(dtype)


def _check_array(array, expected_dtype, expected_shape, expected_data_list=None, expected_is_contiguous=True, device=None):
    assert isinstance(expected_dtype, str)
    expected_dtype = xchainer.dtype(expected_dtype)

    assert isinstance(array.dtype, xchainer.dtype)
    assert isinstance(array.shape, tuple)
    assert array.dtype == expected_dtype
    assert array.shape == expected_shape
    assert array.itemsize == expected_dtype.itemsize
    assert array.size == _size(expected_shape)
    assert array.nbytes == expected_dtype.itemsize * _size(expected_shape)
    if expected_data_list is not None:
        assert array._debug_flat_data == expected_data_list
    assert array.is_contiguous == expected_is_contiguous
    assert array.offset == 0
    if device is None:
        device = xchainer.get_default_device()
    elif isinstance(device, str):
        device = xchainer.get_device(device)
    assert array.device is device


def _check_arrays_equal(array_a, array_b):
    assert array_a.dtype == array_b.dtype
    assert array_a.shape == array_b.shape
    assert array_a.itemsize == array_b.itemsize
    assert array_a.size == array_b.size
    assert array_a.nbytes == array_b.nbytes
    assert array_a._debug_flat_data == array_b._debug_flat_data


def _check_arrays_equal_copy(array_a, array_b):
    _check_arrays_equal(array_a, array_b)
    assert array_b.is_contiguous
    assert 0 == array_b.offset

    # Check memory addresses only if >0 bytes are allocated
    if array_a.size > 0:
        assert array_a._debug_data_memory_address != array_b._debug_data_memory_address


def _check_array_equals_ndarray(array, ndarray, skip_is_contiguous=True):
    assert array.shape == ndarray.shape
    assert array.size == ndarray.size
    assert array.ndim == ndarray.ndim
    assert array.itemsize == ndarray.itemsize
    assert array.nbytes == ndarray.itemsize * ndarray.size
    xchainer.testing.assert_array_equal_ex(array, ndarray)
    if not skip_is_contiguous:
        assert array.is_contiguous == ndarray.flags['C_CONTIGUOUS']


def _check_ndarray_equal_ndarray(ndarray1, ndarray2, skip_strides=False, skip_flags=False):
    assert ndarray1.dtype == ndarray2.dtype
    assert ndarray1.shape == ndarray2.shape
    assert ndarray1.size == ndarray2.size
    assert ndarray1.ndim == ndarray2.ndim
    assert ndarray1.itemsize == ndarray2.itemsize
    assert numpy.array_equal(ndarray1, ndarray2)

    if not skip_strides:
        assert ndarray1.strides == ndarray2.strides
    if not skip_flags:
        assert ndarray1.flags == ndarray2.flags


def _size(shape):
    return functools.reduce(operator.mul, shape, 1)


# Ignores the device argument if with_device is False.
def _check_init(shape, dtype_spec, device=None, with_device=True):
    if with_device:
        array = xchainer.ndarray(shape, dtype_spec, device)
    else:
        array = xchainer.ndarray(shape, dtype_spec)
    expected_dtype = xchainer.dtype(dtype_spec).name
    _check_array(array, expected_dtype, shape, device=device)


@xchainer.testing.parametrize_dtype_specifier('dtype_spec')
def test_init_shape_dtype(shape, dtype_spec):
    _check_init(shape, dtype_spec, with_device=False)


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
@xchainer.testing.parametrize_dtype_specifier('dtype_spec')
def test_init_shape_dtype_device(shape, dtype_spec, device):
    _check_init(shape, dtype_spec, device=device)


# Checks the constructor of ndarray taking a Python list.
# TODO(hvy): This interface differs from numpy.ndarray and should be removed.
@xchainer.testing.parametrize_dtype_specifier('dtype_spec')
def test_init_data_list(shape, dtype_spec):
    data_list = _create_dummy_data(shape, xchainer.dtype(dtype_spec).name)
    expected_dtype = xchainer.dtype(dtype_spec).name
    _check_array(xchainer.ndarray(shape, dtype_spec, data_list), expected_dtype, shape)
    _check_array(xchainer.ndarray(shape, dtype_spec, data_list, 'native:1'), expected_dtype, shape, device='native:1')


def test_to_device():
    a = xchainer.ones((2,), xchainer.float32, device="native:0")
    dst_device = xchainer.get_device("native:1")

    b0 = a.to_device(dst_device)  # by device instance
    assert b0.device is dst_device
    _check_arrays_equal(a, b0)

    b1 = a.to_device("native:1")  # by device name
    assert b1.device is dst_device
    _check_arrays_equal(a, b1)

    b2 = a.to_device("native", 1)  # by backend name and index
    assert b2.device is dst_device
    _check_arrays_equal(a, b2)


def _check_tonumpy(a_np, a_xc):
    xchainer.testing.assert_array_equal_ex(a_xc, a_np, strides_check=False)
    if a_np.size > 0:
        # test buffer is not shared
        a_np.fill(1)
        assert not numpy.array_equal(a_np, xchainer.tonumpy(a_xc))


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_tonumpy(shape, dtype, device):
    a_xc = xchainer.arange(_size(shape)).reshape(shape).astype(dtype)
    a_np = xchainer.tonumpy(a_xc)
    _check_tonumpy(a_np, a_xc)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_tonumpy_non_contiguous(shape, dtype, device):
    a_xc = xchainer.arange(_size(shape)).reshape(shape).astype(dtype).T
    a_np = xchainer.tonumpy(a_xc)
    _check_tonumpy(a_np, a_xc)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_tonumpy_positive_offset(device):
    a_xc = xchainer.arange(6).reshape(2, 3)[:, 1:]
    a_np = xchainer.tonumpy(a_xc)
    _check_tonumpy(a_np, a_xc)


def test_view(shape, dtype):
    data_list = _create_dummy_data(shape, dtype, pattern=1)

    array = xchainer.ndarray(shape, dtype, data_list)
    view = array.view()

    _check_array(view, dtype, shape, data_list)

    # inplace modification
    if len(data_list) > 0:
        array += array
        assert array._debug_flat_data == view._debug_flat_data


def test_view_must_not_share_properties():
    array = xchainer.ndarray((1,), xchainer.float32, [3.0])
    view = array.view()
    # Test preconditions
    assert not array.is_grad_required()
    assert not view.is_grad_required()

    array.require_grad()
    assert not view.is_grad_required(), 'A view must not share is_grad_required with the original array.'


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('copy', [False, True])
# TODO(beam2d): use fixtures.
@pytest.mark.parametrize('src_dtype', ['bool_', 'uint8', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64'])
@pytest.mark.parametrize('dst_dtype', ['bool_', 'uint8', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64'])
def test_astype(xp, shape, device, copy, src_dtype, dst_dtype):
    ndarray = _create_dummy_ndarray(shape, src_dtype)
    a = xp.array(ndarray)
    b = a.astype(dst_dtype, copy=copy)
    assert a is b if src_dtype == dst_dtype and not copy else a is not b
    return b


def test_as_constant_copy(shape, dtype):
    data_list = _create_dummy_data(shape, dtype)

    # Stop gradients on all graphs
    a = xchainer.ndarray(shape, dtype, data_list)
    a.require_grad('graph_1')
    a.require_grad('graph_2')
    assert a.is_grad_required('graph_1')
    assert a.is_grad_required('graph_2')
    b = a.as_constant(copy=True)

    _check_arrays_equal_copy(a, b)
    assert not b.is_grad_required('graph_1')
    assert not b.is_grad_required('graph_2')

    assert a.is_grad_required('graph_1')
    assert a.is_grad_required('graph_2')

    # Stop gradients on some graphs
    a = xchainer.ndarray(shape, dtype, data_list)
    a.require_grad('graph_1')
    a.require_grad('graph_2')
    a.require_grad('graph_3')
    assert a.is_grad_required('graph_1')
    assert a.is_grad_required('graph_2')
    assert a.is_grad_required('graph_3')
    b = a.as_constant(['graph_1', 'graph_2'], copy=True)

    _check_arrays_equal_copy(a, b)
    assert not b.is_grad_required('graph_1')
    assert not b.is_grad_required('graph_2')
    assert b.is_grad_required('graph_3')

    assert a.is_grad_required('graph_1')
    assert a.is_grad_required('graph_2')
    assert a.is_grad_required('graph_3')


def test_as_constant_view(shape, dtype):
    data_list = _create_dummy_data(shape, dtype)

    # Stop gradients on all graphs
    a = xchainer.ndarray(shape, dtype, data_list)
    a.require_grad('graph_1')
    a.require_grad('graph_2')
    assert a.is_grad_required('graph_1')
    assert a.is_grad_required('graph_2')
    b = a.as_constant(copy=False)

    _check_array(b, dtype, shape, data_list)
    assert not b.is_grad_required('graph_1')
    assert not b.is_grad_required('graph_2')

    assert a.is_grad_required('graph_1')
    assert a.is_grad_required('graph_2')

    # Stop gradients on some graphs
    a = xchainer.ndarray(shape, dtype, data_list)
    a.require_grad('graph_1')
    a.require_grad('graph_2')
    a.require_grad('graph_3')
    assert a.is_grad_required('graph_1')
    assert a.is_grad_required('graph_2')
    assert a.is_grad_required('graph_3')
    b = a.as_constant(['graph_1', 'graph_2'], copy=False)

    _check_array(b, dtype, shape, data_list)
    assert not b.is_grad_required('graph_1')
    assert not b.is_grad_required('graph_2')
    assert b.is_grad_required('graph_3')

    assert a.is_grad_required('graph_1')
    assert a.is_grad_required('graph_2')
    assert a.is_grad_required('graph_3')


def test_array_init_invalid_length():
    with pytest.raises(xchainer.DimensionError):
        xchainer.ndarray((), xchainer.int8, [])

    with pytest.raises(xchainer.DimensionError):
        xchainer.ndarray((), xchainer.int8, [1, 1])

    with pytest.raises(xchainer.DimensionError):
        xchainer.ndarray((1,), xchainer.int8, [])

    with pytest.raises(xchainer.DimensionError):
        xchainer.ndarray((1,), xchainer.int8, [1, 1])

    with pytest.raises(xchainer.DimensionError):
        xchainer.ndarray((0,), xchainer.int8, [1])

    with pytest.raises(xchainer.DimensionError):
        xchainer.ndarray((3, 2), xchainer.int8, [1, 1, 1, 1, 1])

    with pytest.raises(xchainer.DimensionError):
        xchainer.ndarray((3, 2), xchainer.int8, [1, 1, 1, 1, 1, 1, 1])


def test_array_repr():
    array = xchainer.ndarray((0,), xchainer.bool_, [])
    assert "array([], shape=(0,), dtype=bool, device='native:0')" == str(array)

    array = xchainer.ndarray((1,), xchainer.bool_, [False])
    assert "array([False], shape=(1,), dtype=bool, device='native:0')" == str(array)

    array = xchainer.ndarray((2, 3), xchainer.int8, [0, 1, 2, 3, 4, 5])
    assert ("array([[0, 1, 2],\n"
            "       [3, 4, 5]], shape=(2, 3), dtype=int8, device='native:0')") == str(array)

    array = xchainer.ndarray((2, 3), xchainer.float32, [0, 1, 2, 3.25, 4, 5])
    assert ("array([[0.  , 1.  , 2.  ],\n"
            "       [3.25, 4.  , 5.  ]], shape=(2, 3), dtype=float32, device='native:0')") == str(array)


def test_array_require_grad():
    array = xchainer.ndarray((3, 1), xchainer.int8, [1, 1, 1])

    assert not array.is_grad_required()
    array.require_grad()
    assert array.is_grad_required()

    with pytest.raises(xchainer.XchainerError):
        array.require_grad()


def test_array_require_grad_with_graph_id():
    array = xchainer.ndarray((3, 1), xchainer.int8, [1, 1, 1])

    assert not array.is_grad_required('graph_1')
    array.require_grad('graph_1')
    assert array.is_grad_required('graph_1')
    with pytest.raises(xchainer.XchainerError):
        array.require_grad('graph_1')

    # keyword arguments
    assert not array.is_grad_required(graph_id='graph_2')
    array.require_grad(graph_id='graph_2')
    assert array.is_grad_required('graph_2')
    assert array.is_grad_required(graph_id='graph_2')
    with pytest.raises(xchainer.XchainerError):
        array.require_grad(graph_id='graph_2')

    # Raise TypeError if given graph_id is None
    with pytest.raises(TypeError):
        array.require_grad(None)
    with pytest.raises(TypeError):
        array.is_grad_required(None)


def test_array_grad():
    array = xchainer.ndarray((3, 1), xchainer.int8, [1, 1, 1])
    grad = xchainer.ndarray((3, 1), xchainer.float32, [0.5, 0.5, 0.5])

    with pytest.raises(xchainer.XchainerError):
        array.get_grad()
    with pytest.raises(xchainer.XchainerError):
        array.set_grad(grad)
    with pytest.raises(xchainer.XchainerError):
        array.cleargrad()

    # Gradient methods
    array.require_grad().set_grad(grad)
    assert array.get_grad() is not None
    assert array.get_grad()._debug_flat_data == grad._debug_flat_data

    array.cleargrad()  # clear
    assert array.get_grad() is None

    array.set_grad(grad)
    assert array.get_grad() is not None
    assert array.get_grad()._debug_flat_data == grad._debug_flat_data

    array.set_grad(None)  # clear
    assert array.get_grad() is None

    # Gradient attributes
    array.grad = grad
    assert array.get_grad() is not None
    assert array.get_grad() is array.grad

    array.grad = None  # clear
    assert array.get_grad() is None


def test_array_grad_with_graph_id():
    array = xchainer.ndarray((3, 1), xchainer.int8, [1, 1, 1])
    grad = xchainer.ndarray((3, 1), xchainer.float32, [0.5, 0.5, 0.5])

    with pytest.raises(xchainer.XchainerError):
        array.get_grad('graph_1')
    with pytest.raises(xchainer.XchainerError):
        array.set_grad(grad, 'graph_1')
    with pytest.raises(xchainer.XchainerError):
        array.cleargrad('graph_1')

    array.require_grad('graph_1').set_grad(grad, 'graph_1')
    assert array.get_grad('graph_1') is not None
    assert array.get_grad('graph_1')._debug_flat_data == grad._debug_flat_data

    array.cleargrad('graph_1')  # clear
    assert array.get_grad('graph_1') is None

    # keyword arguments
    with pytest.raises(xchainer.XchainerError):
        array.get_grad(graph_id='graph_2')
    with pytest.raises(xchainer.XchainerError):
        array.set_grad(grad, graph_id='graph_2')
    with pytest.raises(xchainer.XchainerError):
        array.cleargrad(graph_id='graph_2')

    array.require_grad(graph_id='graph_2').set_grad(grad, graph_id='graph_2')
    assert array.get_grad('graph_2') is not None
    assert array.get_grad(graph_id='graph_2') is not None
    assert array.get_grad('graph_2')._debug_flat_data == grad._debug_flat_data
    assert array.get_grad(graph_id='graph_2')._debug_flat_data == grad._debug_flat_data

    array.cleargrad(graph_id='graph_2')  # clear
    assert array.get_grad('graph_2') is None
    assert array.get_grad(graph_id='graph_2') is None

    # Raise TypeError if given graph_id is None
    with pytest.raises(TypeError):
        array.get_grad(None)
    with pytest.raises(TypeError):
        array.set_grad(grad, None)


def test_array_grad_no_deepcopy():
    shape = (3, 1)
    dtype = xchainer.int8
    array = xchainer.ndarray(shape, dtype, [2, 5, 1])
    grad = xchainer.ndarray(shape, dtype, [5, 7, 8])

    # Set grad
    array.require_grad().set_grad(grad)

    # Retrieve grad twice and assert they share the same underlying data
    grad1 = array.get_grad()
    grad2 = array.get_grad()

    grad1 *= xchainer.ndarray(shape, dtype, [2, 2, 2])
    assert grad2._debug_flat_data == [10, 14, 16], 'grad getter must not incur a copy'


def test_array_cleargrad():
    shape = (3, 1)
    dtype = xchainer.int8
    array = xchainer.ndarray(shape, dtype, [2, 5, 1])
    grad = xchainer.ndarray(shape, dtype, [5, 7, 8])

    # Set grad, get it and save it
    array.require_grad().set_grad(grad)
    del grad
    saved_grad = array.get_grad()

    # Clear grad
    array.cleargrad()
    assert array.get_grad() is None

    assert saved_grad._debug_flat_data == [5, 7, 8], 'Clearing grad must not affect previously retrieved grad'


def test_array_grad_identity():
    shape = (3, 1)
    array = xchainer.ndarray(shape, xchainer.int8, [1, 1, 1])
    grad = xchainer.ndarray(shape, xchainer.float32, [0.5, 0.5, 0.5])
    array.require_grad().set_grad(grad)

    assert array.get_grad() is grad, 'grad must preserve physical identity'
    assert array.get_grad() is grad, 'grad must preserve physical identity in repeated retrieval'

    # array.grad and grad share the same data
    grad += xchainer.ndarray(shape, xchainer.float32, [2, 2, 2])
    assert array.get_grad()._debug_flat_data == [2.5, 2.5, 2.5], 'A modification to grad must affect array.grad'

    array_grad = array.get_grad()
    array_grad += xchainer.ndarray(shape, xchainer.float32, [1, 1, 1])
    assert grad._debug_flat_data == [3.5, 3.5, 3.5], 'A modification to array.grad must affect grad'


def test_array_require_grad_multiple_graphs_forward():
    x1 = xchainer.ndarray((3, 1), xchainer.int8, [1, 1, 1])
    x2 = xchainer.ndarray((3, 1), xchainer.int8, [1, 1, 1])

    graph_id1 = 'graph_1'
    graph_id2 = 'graph_2'

    x1.require_grad(graph_id1)
    x2.require_grad(graph_id2)

    assert x1.is_grad_required(graph_id1)
    assert x2.is_grad_required(graph_id2)

    assert not x1.is_grad_required(graph_id2)
    assert not x2.is_grad_required(graph_id1)

    y = x1 * x2

    assert y.is_grad_required(graph_id1)
    assert y.is_grad_required(graph_id2)

    # No unspecified graphs are generated
    assert not y.is_grad_required(xchainer.DEFAULT_GRAPH_ID)
    assert not y.is_grad_required('graph_3')


def test_array_backward():
    x1 = xchainer.ndarray((3, 1), xchainer.int8, [1, 1, 1]).require_grad(graph_id='graph_1')
    x2 = xchainer.ndarray((3, 1), xchainer.int8, [1, 1, 1]).require_grad(graph_id='graph_1')
    y = x1 * x2

    y.backward(graph_id='graph_1', enable_double_backprop=True)
    gx1 = x1.get_grad(graph_id='graph_1')
    x1.set_grad(None, graph_id='graph_1')

    gx1.backward(graph_id='graph_1')
    assert gx1.get_grad(graph_id='graph_1') is not None


# TODO(niboshi): Remove strides_check=False
@xchainer.testing.numpy_xchainer_array_equal(strides_check=False)
@pytest.mark.parametrize("shape,indices", [
    # empty indexing
    ((), ()),
    ((3,), ()),
    ((2, 2, 2), ()),
    # integer indexing - non-tuple indexing
    ((3,), 0),
    ((3,), 1),
    ((3,), 2),
    ((3,), -1),
    ((2, 3), 0),
    ((2, 3), 1),
    # integer indexining - tuple indexing
    ((3,), (0,)),
    ((3,), (1,)),
    ((3,), (2,)),
    ((3,), (-1,)),
    ((2, 3), (0,)),
    ((2, 3), (1,)),
    ((2, 3), (0, 0)),
    ((2, 3), (1, 1)),
    ((2, 3, 4), (0, -2, 3)),
    ((2, 3, 4), (1, 0)),
    # slice indexing - non-tuple indexing
    ((3,), slice(None)),
    ((3,), slice(2)),
    ((3,), slice(0, 3)),
    ((3,), slice(0, 2)),
    ((3,), slice(1, 3)),
    ((3,), slice(0, 0)),
    ((3,), slice(0, 1)),
    ((3,), slice(2, 0, -1)),
    ((3,), slice(-2, -1)),
    ((3,), slice(2, None, -1)),
    ((3,), slice(None, 0, 1)),
    ((3,), slice(None, -1, -1)),
    ((3,), slice(None, -2, -1)),
    ((6,), slice(0, 6, 2)),
    ((6,), slice(1, 6, 2)),
    ((6,), slice(5, None, -2)),
    # slice indexing - tuple indexing
    ((3,), (slice(None),)),
    ((3,), (slice(2),)),
    ((3,), (slice(0, 3),)),
    ((3,), (slice(0, 2),)),
    ((3,), (slice(1, 3),)),
    ((3,), (slice(0, 0),)),
    ((3,), (slice(0, 1),)),
    ((3,), (slice(2, 0, -1),)),
    ((3,), (slice(-2, -1),)),
    ((3,), (slice(2, None, -1),)),
    ((3,), (slice(None, 0, 1),)),
    ((3,), (slice(None, -1, -1),)),
    ((3,), (slice(None, -2, -1),)),
    ((6,), (slice(0, 6, 2),)),
    ((6,), (slice(1, 6, 2),)),
    ((6,), (slice(5, None, -2),)),
    ((2, 3), (slice(None), slice(None))),
    ((2, 3), (slice(1), slice(2))),
    ((2, 3), (slice(0, 2), slice(0, 3))),
    ((2, 3), (slice(0, 2), slice(0, -1))),
    ((2, 3), (slice(0, None, -1), slice(2, 3))),
    ((2, 3), (slice(0, None, None), slice(-2, 0, -1))),
    ((2, 3), (slice(1, 2), slice(0, 2))),
    ((2, 3), (slice(-2, None, -1), slice(0, 3))),
    ((2, 3), (slice(-2, None, -1), slice(-3, None, -1))),
    ((2, 3), (slice(-2, None, -1), slice(None, None, -2))),
    ((2, 3), (slice(1, 2), slice(None, None, 1))),
    ((2, 3), (slice(1, 2), slice(None, None, 2))),
    ((2, 3, 4), (slice(1), slice(-2, 3), slice(1, None, -1))),
    # newaxis indexing - non-tuple indexing
    ((), xchainer.newaxis),
    ((3,), xchainer.newaxis),
    # newaxis indexing - tuple indexing
    ((), (xchainer.newaxis,)),
    ((3,), (xchainer.newaxis,)),
    ((2, 3), (xchainer.newaxis, xchainer.newaxis)),
    # mixed indexing - tuple indexing
    ((2, 3), (0, slice(1, 3))),
    ((4, 3), (slice(1, 3), 1)),
    ((2, 3, 4), (1, slice(2,), slice(1, 3))),
    ((2, 3), (1, xchainer.newaxis, slice(1, 3))),
    ((2, 3, 4), (slice(0, 1), slice(1, 2), slice(1, 3), xchainer.newaxis)),
    ((2, 3, 4), (slice(0, 1), slice(1, 2), xchainer.newaxis, slice(1, 3))),
    ((2, 3, 4), (slice(0, 1), xchainer.newaxis, slice(1, 2), slice(1, 3))),
    ((2, 3, 4), (xchainer.newaxis, slice(0, 1), slice(1, 2), slice(1, 3))),
    ((2, 3, 4), (1, slice(2,), xchainer.newaxis, slice(1, 3), xchainer.newaxis)),
])
def test_getitem(xp, shape, indices):
    ndarray = _create_dummy_ndarray(shape, 'int32')
    a = xp.array(ndarray)
    return a[indices]


# TODO(hvy): Add cases where axis=None, when supported.
# TODO(hvy): Add cases where indices is not int64, when supported.
# shape,indices,axis
_take_valid_params = [
    ((3,), [0], 0),
    ((3,), [1], 0),
    ((2, 3), [0], 0),
    ((2, 3), [0], 1),
    ((2, 3), [0], -1),
    ((2, 3), [1], 0),
    ((2, 3), [0, -1], 0),
    ((2, 3), [1, 0], 0),
    ((2, 3), [1, 2], 1),
    ((2, 3), [2, 1], 1),
    ((2, 3), [[0], [1]], 0),
]

_take_invalid_params = [
    # Axis out of bounds
    ((2, 3), [0], 2),
    ((2, 3), [0], -3),
]


@xchainer.testing.numpy_xchainer_array_equal(dtype_check=False, accept_error=(xchainer.DimensionError, numpy.AxisError))
@pytest.mark.parametrize("shape,indices,axis", _take_valid_params + _take_invalid_params)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_take(is_module, xp, shape, indices, axis, device):
    a = xp.arange(_size(shape)).reshape(shape)

    # First convert to ndarray since some indices are nested lists which
    # xchainer cannot convert. Additionally, dtype is cast to int64 since no
    # other dtypes are currently supported by xchainer.take
    indices = numpy.array(indices).astype('int64')

    if is_module:
        return xp.take(a, xp.array(indices), axis)
    else:
        return a.take(xp.array(indices), axis)


@pytest.mark.parametrize('is_module', [False, True])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])  # TODO(niboshi): Use float_dtype fixture
@pytest.mark.parametrize('shape,indices,axis', _take_valid_params)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_take_backward(is_module, dtype, shape, indices, axis, device):
    def func(a, indices, axis):
        if is_module:
            return xchainer.take(a, indices, axis)
        else:
            return a.take(indices, axis)

    # First convert to ndarray since some indices are nested lists which
    # xchainer cannot convert. Additionally, dtype is cast to int64 since no
    # other dtypes are currently supported by xchainer.take
    indices = xchainer.array(numpy.array(indices, dtype='int64'))

    a = xchainer.arange(_size(shape)).reshape(shape).astype(dtype).require_grad()
    output_shape = func(a, indices, axis).shape

    numpy.random.seed(0)  # TODO(niboshi): Reconsider the use of random values
    go = xchainer.array(numpy.random.uniform(-1, 1, output_shape).astype(dtype)).require_grad()
    ggi = xchainer.array(numpy.random.uniform(-1, 1, shape).astype(dtype))
    epsi = xchainer.full_like(a, 1e-3)
    epso = xchainer.full_like(go, 1e-3)

    def func_bwd(inputs):
        return func(inputs[0], indices, axis),

    def func_dbwd(inputs):
        y = func(inputs[0], indices, axis)
        return y * y,  # make nonlinear

    xchainer.check_backward(func_bwd, (a,), (go,), (epsi,))
    xchainer.check_double_backward(func_dbwd, (a,), (go,), (ggi,), (epsi, epso))


# TODO(sonots): Fix type compatibility
# TODO(niboshi): Remove strides_check=False
@xchainer.testing.numpy_xchainer_array_equal(dtype_check=False, strides_check=False)
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("shape,axis", [
    ((), None),
    ((), ()),
    ((2,), None),
    ((2,), ()),
    ((2,), 0),
    ((2,), (0,)),
    ((2,), (-1,)),
    ((2, 3), None),
    ((2, 3), ()),
    ((2, 3), 0),
    ((2, 3), (0,)),
    ((2, 3), (1,)),
    ((2, 3), (-1,)),
    ((2, 3), (-2,)),
    ((2, 3), (0, 1)),
    ((2, 3), (-2, -1)),
    ((1, 3), None),  # sum over 1-dim axis
    ((0, 3), None),  # sum over 0-dim axis
    # Sum over axes that are in the middle or apart
    ((2, 3, 4), (1,)),
    ((2, 3, 4), (0, 2)),
    # Sum over axes that are apart and/or unsorted
    ((2, 3), (1, 0)),
    ((2, 3, 4), (2, 0)),
    ((2, 3, 4), (2, 0, 1)),
    ((2, 3, 4), (-2, 2, 0)),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_sum(is_module, xp, device, shape, axis, keepdims):
    ndarray = _create_dummy_ndarray(shape, 'int32')
    a = xp.array(ndarray)
    if is_module:
        return xp.sum(a, axis=axis, keepdims=keepdims)
    else:
        return a.sum(axis=axis, keepdims=keepdims)


@xchainer.testing.numpy_xchainer_array_equal(accept_error=(xchainer.DimensionError, ValueError))
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("shape,axis", [
    # ((), 0), # TODO(sonots): Fix compatibility
    ((), 1),
    ((), (1,)),
    ((2,), 2),
    ((2,), (2,)),
    ((2,), (-2,)),
    ((2, 3,), (-3,)),
    ((2, 3,), (-3, -4)),
    ((2, 3,), (0, 0)),
    ((2, 3,), (-1, -1)),
    ((2, 3,), (0, 1, 1)),
    ((2, 3,), (0, -2)),
])
def test_invalid_sum(is_module, xp, shape, axis, keepdims):
    ndarray = _create_dummy_ndarray(shape, 'int32')
    a = xp.array(ndarray)
    if is_module:
        xp.sum(a, axis=axis, keepdims=keepdims)
    else:
        a.sum(axis=axis, keepdims=keepdims)


# TODO(sonots): Fix type compatibility for when shape is ()
@xchainer.testing.numpy_xchainer_array_equal(dtype_check=False)
@pytest.mark.parametrize("shape,value", [
    ((), -1),
    ((), 1),
    ((1,), -1),
    ((1,), 1),
    ((2,), 1),
    ((2, 3), 3),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_maximum_with_scalar(xp, device, shape, value, signed_dtype):
    ndarray = _create_dummy_ndarray(shape, signed_dtype)
    a = xp.array(ndarray)
    return xp.maximum(a, value)


def _create_dummy_array_for_dot(xp, shape, dtype):
    x = numpy.arange(numpy.prod(shape)).reshape(shape)
    if dtype == 'bool_':
        x = numpy.asarray(x % 2 == 0)
    else:
        x = x.astype(dtype)
    return xp.array(x)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('a_shape,b_shape', [
    ((), ()),
    ((), (2, 3)),
    ((2, 0), (0, 3)),
    ((0, 0), (0, 0)),
    ((2, 3), (3, 4)),
    # TODO(niboshi): Add test cases for more than 2 ndim
])
# TODO(niboshi): Add 'cuda:0'
@pytest.mark.parametrize_device(['native:0'])
def test_dot(is_module, xp, device, a_shape, b_shape, dtype):
    a = _create_dummy_array_for_dot(xp, a_shape, dtype)
    b = _create_dummy_array_for_dot(xp, b_shape, dtype)
    if is_module:
        return xp.dot(a, b)
    else:
        return a.dot(b)


@xchainer.testing.numpy_xchainer_array_equal(accept_error=(xchainer.DimensionError, ValueError))
@pytest.mark.parametrize('a_shape,b_shape', [
    ((3, 2), (1, 3)),
])
# TODO(niboshi): Add 'cuda:0'
@pytest.mark.parametrize_device(['native:0'])
def test_invalid_dot(is_module, xp, device, a_shape, b_shape, dtype):
    a = _create_dummy_array_for_dot(xp, a_shape, dtype)
    b = _create_dummy_array_for_dot(xp, b_shape, dtype)
    if is_module:
        return xp.dot(a, b)
    else:
        return a.dot(b)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('value', [-1, 0, 1, 2, 2.3, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_fill(xp, shape, dtype, value, device):
    a = xp.empty(shape, dtype)
    a.fill(value)
    return a


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('value', [-1, 0, 1, 2, 2.3, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_fill_with_scalar(xp, device, shape, dtype, value):
    a = xp.empty(shape, dtype)
    if xp is xchainer:
        value = xchainer.Scalar(value, dtype)
    a.fill(value)
    return a


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0), numpy.asarray(-4), numpy.asarray(4),
    numpy.asarray(-float('inf')), numpy.asarray(float('inf')), numpy.asarray(float('nan')),
    numpy.full((), 2), numpy.full((0,), 2), numpy.full((2, 3), 2)
])
# TODO(niboshi): Dtype promotion is not supported yet.
@xchainer.testing.numpy_xchainer_array_equal()
def test_exp(xp, device, input, float_dtype):
    dtype = float_dtype
    a = xp.array(input.astype(dtype))
    return xp.exp(a)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0), numpy.asarray(-1), numpy.asarray(1), numpy.asarray(10), numpy.asarray(float('inf')), numpy.asarray(float('nan')),
    numpy.full((), 2), numpy.full((0,), 2), numpy.full((2, 3), 2)
])
# TODO(niboshi): Dtype promotion is not supported yet.
@xchainer.testing.numpy_xchainer_array_equal()
def test_log(xp, device, input, float_dtype):
    dtype = float_dtype
    a = xp.array(input.astype(dtype))
    return xp.log(a)


_logsumexp_params = [
    ((2,), 0),
    ((2,), -1),
    ((2, 3), None),
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 3), -2),
    ((2, 3), (0, 1)),
    ((2, 3), (-2, 1)),
    ((1, 2, 3), None),
    ((1, 2, 3), (1)),
    ((1, 2, 3), (1, 0)),
    ((1, 2, 3), (0, 1, 2)),
]


_invalid_logsumexp_params = [
    # Axis out of bounds
    ((2,), 1),
    ((2,), -2),
    ((2,), (0, 1)),
    ((2, 3), (0, 1, 2)),
    # Duplicate axes
    ((2,), (0, 0)),
    ((2, 3), (0, 0)),
]


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _logsumexp_params)
@pytest.mark.parametrize('keepdims', [True, False])
@xchainer.testing.numpy_xchainer_allclose(rtol=1e-7, atol=0, dtype_check=False)
# TODO(hvy): Dtype promotion is not supported yet.
def test_logsumexp(xp, device, a_shape, axis, float_dtype, keepdims):
    a = xp.arange(_size(a_shape), dtype=float_dtype).reshape(a_shape)
    if xp is numpy:
        return xp.log(xp.sum(xp.exp(a), axis=axis, keepdims=keepdims))
    return xp.logsumexp(a, axis=axis, keepdims=keepdims)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _invalid_logsumexp_params)
@pytest.mark.parametrize('keepdims', [True, False])
# TODO(hvy): Dtype promotion is not supported yet.
# TODO(hvy): Should not overflow for large numbers, add tests
def test_invalid_logsumexp(device, a_shape, axis, float_dtype, keepdims):
    a = xchainer.arange(_size(a_shape), dtype=float_dtype).reshape(a_shape)
    with pytest.raises(xchainer.DimensionError):
        xchainer.logsumexp(a, axis=axis, keepdims=keepdims)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _logsumexp_params)
@xchainer.testing.numpy_xchainer_allclose(rtol=1e-7, atol=1e-5, dtype_check=False)
# TODO(hvy): Dtype promotion is not supported yet.
def test_log_softmax(xp, device, a_shape, axis, float_dtype):
    a = xp.arange(_size(a_shape), dtype=float_dtype).reshape(a_shape)
    if xp is numpy:
        # Default is the second axis
        axis = axis if axis is not None else 1
        return a - xp.log(xp.sum(xp.exp(a), axis=axis, keepdims=True))
    return xp.log_softmax(a, axis=axis)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _invalid_logsumexp_params)
# TODO(hvy): Dtype promotion is not supported yet.
def test_invalid_log_softmax(device, a_shape, axis, float_dtype):
    a = xchainer.arange(_size(a_shape), dtype=float_dtype).reshape(a_shape)
    with pytest.raises(xchainer.DimensionError):
        return xchainer.log_softmax(a, axis=axis)


_min_max_single_axis_params = [
    # input, axis
    # valid params
    (numpy.asarray(0), None),
    (numpy.asarray(-1), None),
    (numpy.asarray(float('inf')), None),
    (numpy.asarray(float('nan')), None),
    (numpy.asarray(-float('inf')), None),
    (numpy.asarray([4, 1, 4, 1]), None),
    (numpy.asarray([4, 1, 4, 1]), 0),
    (numpy.asarray([[4, 4, 1, 1], [4, 1, 4, 1]]), 0),
    (numpy.asarray([[4, 4, 1, 1], [4, 1, 4, 1]]).T, 1),
    (numpy.asarray([-0.0, +0.0, +0.0, -0.0]), None),
    (numpy.asarray([[True, True, False, False], [True, False, True, False]]), 0),
    (numpy.ones((2, 0, 3)), 2),
    (numpy.ones((2, 3)), 1),
    (numpy.ones((2, 3)), -2),
    # invalid params
    (numpy.ones((0,)), None),
    (numpy.ones((2, 0, 3)), 1),
    (numpy.ones((2, 0, 3)), None),
    (numpy.ones((2, 3)), 2),
    (numpy.ones((2, 3)), -3),
]


@pytest.mark.parametrize('input,axis', _min_max_single_axis_params)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal(accept_error=(ValueError, xchainer.DimensionError))
def test_argmax(is_module, xp, device, input, axis, dtype):
    try:
        a_np = input.astype(dtype)
    except (ValueError, OverflowError):
        return xp.zeros(())  # invalid combination of data and dtype

    a = xp.array(a_np)
    if is_module:
        return xp.argmax(a, axis)
    else:
        return a.argmax(axis)


_min_max_multi_axis_params = _min_max_single_axis_params + [
    # input, axis
    # valid params
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (0, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (-2, -1)),
    # invalid params
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (-3, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 2)),
]


def test_max_amax():
    assert xchainer.amax is xchainer.max


@pytest.mark.parametrize('input,axis', _min_max_multi_axis_params)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
# TODO(niboshi): Remove strides_check=False
@xchainer.testing.numpy_xchainer_array_equal(accept_error=(ValueError, xchainer.DimensionError), strides_check=False)
def test_max(is_module, xp, device, input, axis, dtype):
    try:
        a_np = input.astype(dtype)
    except (ValueError, OverflowError):
        return xp.zeros(())  # invalid combination of data and dtype

    a = xp.array(a_np)
    if is_module:
        return xp.max(a, axis)
    else:
        return a.max(axis)
