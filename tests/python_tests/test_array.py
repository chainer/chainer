import numpy
import pytest

import xchainer
import xchainer.testing

from tests import array_utils


def _check_array(array, expected_dtype, expected_shape, expected_data_list=None, device=None):
    expected_dtype = xchainer.dtype(expected_dtype)

    assert isinstance(array.dtype, xchainer.dtype)
    assert isinstance(array.shape, tuple)
    assert array.dtype == expected_dtype
    assert array.shape == expected_shape
    assert array.itemsize == expected_dtype.itemsize
    assert array.size == array_utils.total_size(expected_shape)
    assert array.nbytes == expected_dtype.itemsize * array_utils.total_size(expected_shape)
    if expected_data_list is not None:
        assert array._debug_flat_data == expected_data_list

    assert array.is_contiguous

    array_utils.check_device(array, device)


@xchainer.testing.parametrize_dtype_specifier('dtype_spec')
def test_init(shape, dtype_spec):
    array = xchainer.ndarray(shape, dtype_spec)
    _check_array(array, dtype_spec, shape)


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
@xchainer.testing.parametrize_dtype_specifier('dtype_spec')
def test_init_with_device(shape, dtype_spec, device):
    array = xchainer.ndarray(shape, dtype_spec, device=device)
    _check_array(array, dtype_spec, shape, device=device)


# Checks the constructor of ndarray taking a Python list.
# TODO(hvy): This interface differs from numpy.ndarray and should be removed.
@xchainer.testing.parametrize_dtype_specifier('dtype_spec')
def test_init_from_list(shape, dtype_spec):
    dtype_name = xchainer.dtype(dtype_spec).name
    data_list = array_utils.create_dummy_ndarray(numpy, shape, dtype_name).ravel().tolist()
    array = xchainer.ndarray(shape, dtype_spec, data_list)
    _check_array(array, dtype_name, shape, data_list)


@xchainer.testing.parametrize_dtype_specifier('dtype_spec')
@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
def test_init_from_list_with_device(shape, dtype_spec, device):
    dtype_name = xchainer.dtype(dtype_spec).name
    data_list = array_utils.create_dummy_ndarray(numpy, shape, dtype_name).ravel().tolist()
    array = xchainer.ndarray(shape, dtype_spec, data_list, device)
    _check_array(array, dtype_name, shape, data_list, device=device)


def test_init_invalid_length():
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


def test_to_device():
    a = xchainer.ones((2,), xchainer.float32, device="native:0")
    dst_device = xchainer.get_device("native:1")

    b0 = a.to_device(dst_device)  # by device instance
    assert b0.device is dst_device
    xchainer.testing.assert_array_equal_ex(a, b0)

    b1 = a.to_device("native:1")  # by device name
    assert b1.device is dst_device
    xchainer.testing.assert_array_equal_ex(a, b1)

    b2 = a.to_device("native", 1)  # by backend name and index
    assert b2.device is dst_device
    xchainer.testing.assert_array_equal_ex(a, b2)


def _check_tonumpy(a_np, a_xc):
    xchainer.testing.assert_array_equal_ex(a_xc, a_np, strides_check=False)
    if a_np.size > 0:
        # test buffer is not shared
        a_np.fill(1)
        assert not numpy.array_equal(a_np, xchainer.tonumpy(a_xc))


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_tonumpy(shape, dtype, device):
    a_xc = array_utils.create_dummy_ndarray(xchainer, shape, dtype)
    a_np = xchainer.tonumpy(a_xc)
    _check_tonumpy(a_np, a_xc)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_tonumpy_non_contiguous(shape, dtype, device):
    a_xc = array_utils.create_dummy_ndarray(xchainer, shape, dtype).T
    a_np = xchainer.tonumpy(a_xc)
    _check_tonumpy(a_np, a_xc)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_tonumpy_positive_offset(device):
    a_xc = xchainer.arange(6).reshape(2, 3)[:, 1:]
    a_np = xchainer.tonumpy(a_xc)
    _check_tonumpy(a_np, a_xc)


def test_view(shape, dtype):
    array = array_utils.create_dummy_ndarray(xchainer, shape, dtype)
    view = array.view()

    xchainer.testing.assert_array_equal_ex(view, array)
    assert view.device is xchainer.get_default_device()

    # inplace modification
    if array.size > 0:
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


@xchainer.testing.numpy_xchainer_array_equal(strides_check=False)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('copy', [False, True])
# TODO(beam2d): use fixtures.
@pytest.mark.parametrize('src_dtype', ['bool_', 'uint8', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64'])
@pytest.mark.parametrize('dst_dtype', ['bool_', 'uint8', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64'])
def test_astype(xp, shape, device, copy, src_dtype, dst_dtype):
    a = array_utils.create_dummy_ndarray(xp, shape, src_dtype)

    # Casting negative value to unsigned int behaves different in CUDA
    if device.name == 'cuda:0' and \
            src_dtype in xchainer.testing.signed_dtypes and \
            dst_dtype in xchainer.testing.unsigned_dtypes:
        a = xp.maximum(a, 0)

    b = a.astype(dst_dtype, copy=copy)
    assert a is b if src_dtype == dst_dtype and not copy else a is not b
    return b


def test_as_grad_stopped_copy(shape, dtype):
    def check(array_a, array_b):
        xchainer.testing.assert_array_equal_ex(array_a, array_b, strides_check=False)

        assert array_b.is_contiguous

        # Check memory addresses only if >0 bytes are allocated
        if array_a.size > 0:
            assert array_a._debug_data_memory_address != array_b._debug_data_memory_address

    # Stop gradients on all graphs
    with xchainer.graph_scope('graph_1') as graph_1, \
            xchainer.graph_scope('graph_2') as graph_2, \
            xchainer.graph_scope('graph_3') as graph_3:

        a = array_utils.create_dummy_ndarray(xchainer, shape, dtype)
        a.require_grad(graph_1)
        a.require_grad(graph_2)
        assert a.is_grad_required(graph_1)
        assert a.is_grad_required(graph_2)
        b = a.as_grad_stopped(copy=True)

        check(a, b)
        assert not b.is_grad_required(graph_1)
        assert not b.is_grad_required(graph_2)

        assert a.is_grad_required(graph_1)
        assert a.is_grad_required(graph_2)

        # Stop gradients on some graphs
        a = array_utils.create_dummy_ndarray(xchainer, shape, dtype)
        a.require_grad(graph_1)
        a.require_grad(graph_2)
        a.require_grad(graph_3)
        assert a.is_grad_required(graph_1)
        assert a.is_grad_required(graph_2)
        assert a.is_grad_required(graph_3)
        b = a.as_grad_stopped([graph_1, graph_2], copy=True)

        check(a, b)
        assert not b.is_grad_required(graph_1)
        assert not b.is_grad_required(graph_2)
        assert b.is_grad_required(graph_3)

        assert a.is_grad_required(graph_1)
        assert a.is_grad_required(graph_2)
        assert a.is_grad_required(graph_3)


def test_as_grad_stopped_view(shape, dtype):
    # Stop gradients on all graphs
    with xchainer.graph_scope('graph_1') as graph_1, \
            xchainer.graph_scope('graph_2') as graph_2, \
            xchainer.graph_scope('graph_3') as graph_3:

        a = array_utils.create_dummy_ndarray(xchainer, shape, dtype)
        a.require_grad(graph_1)
        a.require_grad(graph_2)
        assert a.is_grad_required(graph_1)
        assert a.is_grad_required(graph_2)
        b = a.as_grad_stopped(copy=False)

        xchainer.testing.assert_array_equal_ex(a, b)
        assert b.device is a.device
        assert not b.is_grad_required(graph_1)
        assert not b.is_grad_required(graph_2)

        assert a.is_grad_required(graph_1)
        assert a.is_grad_required(graph_2)

        # Stop gradients on some graphs
        a = array_utils.create_dummy_ndarray(xchainer, shape, dtype)
        a.require_grad(graph_1)
        a.require_grad(graph_2)
        a.require_grad(graph_3)
        assert a.is_grad_required(graph_1)
        assert a.is_grad_required(graph_2)
        assert a.is_grad_required(graph_3)
        b = a.as_grad_stopped([graph_1, graph_2], copy=False)

        xchainer.testing.assert_array_equal_ex(a, b)
        assert b.device is a.device
        assert not b.is_grad_required(graph_1)
        assert not b.is_grad_required(graph_2)
        assert b.is_grad_required(graph_3)

        assert a.is_grad_required(graph_1)
        assert a.is_grad_required(graph_2)
        assert a.is_grad_required(graph_3)


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


@pytest.mark.parametrize('graph_args', [(None,), ()])
def test_array_require_grad_without_graph_id(graph_args):
    array = xchainer.ndarray((3, 1), xchainer.int8, [1, 1, 1])

    assert not array.is_grad_required(*graph_args)
    assert not array.is_grad_required(xchainer.anygraph)
    array.require_grad(*graph_args)
    assert array.is_grad_required(*graph_args)
    assert array.is_grad_required(xchainer.anygraph)

    # Repeated calls should not fail, but do nothing
    array.require_grad(*graph_args)
    assert array.is_grad_required(*graph_args)
    assert array.is_grad_required(xchainer.anygraph)


def test_array_require_grad_with_graph_id():
    array = xchainer.ndarray((3, 1), xchainer.int8, [1, 1, 1])

    with xchainer.graph_scope('graph_1') as graph_1:
        assert not array.is_grad_required(graph_1)
        array.require_grad(graph_1)
        assert array.is_grad_required(graph_1)

        # Repeated calls should not fail, but do nothing
        array.require_grad(graph_1)
        assert array.is_grad_required(graph_1)

    # keyword arguments
    with xchainer.graph_scope('graph_2') as graph_2:
        assert not array.is_grad_required(graph_id=graph_2)
        array.require_grad(graph_id=graph_2)
        assert array.is_grad_required(graph_2)
        assert array.is_grad_required(graph_id=graph_2)

        # Repeated calls should not fail, but do nothing
        array.require_grad(graph_id=graph_2)
        assert array.is_grad_required(graph_id=graph_2)


@pytest.mark.parametrize('graph_args', [(None,), ()])
def test_array_grad_without_graph_id(graph_args):
    array = xchainer.ndarray((3, 1), xchainer.float32, [1., 1., 1.])
    grad = xchainer.ndarray((3, 1), xchainer.float32, [0.5, 0.5, 0.5])

    with pytest.raises(xchainer.XchainerError):
        array.get_grad(*graph_args)
    with pytest.raises(xchainer.XchainerError):
        array.set_grad(grad, *graph_args)
    with pytest.raises(xchainer.XchainerError):
        array.cleargrad(*graph_args)

    # Gradient methods
    array.require_grad().set_grad(grad, *graph_args)
    assert array.get_grad(*graph_args) is not None
    assert array.get_grad(*graph_args)._debug_flat_data == grad._debug_flat_data

    array.cleargrad(*graph_args)  # clear
    assert array.get_grad(*graph_args) is None

    array.set_grad(grad, *graph_args)
    assert array.get_grad(*graph_args) is not None
    assert array.get_grad(*graph_args)._debug_flat_data == grad._debug_flat_data

    array.set_grad(None, *graph_args)  # clear
    assert array.get_grad(*graph_args) is None

    # Gradient attributes
    array.grad = grad
    assert array.get_grad(*graph_args) is not None
    assert array.get_grad(*graph_args) is array.grad

    array.grad = None  # clear
    assert array.get_grad(*graph_args) is None


def test_array_grad_with_graph_id():
    array = xchainer.ndarray((3, 1), xchainer.float32, [1., 1., 1.])
    grad = xchainer.ndarray((3, 1), xchainer.float32, [0.5, 0.5, 0.5])

    with xchainer.graph_scope('graph_1') as graph_1:
        with pytest.raises(xchainer.XchainerError):
            array.get_grad(graph_1)
        with pytest.raises(xchainer.XchainerError):
            array.set_grad(grad, graph_1)
        with pytest.raises(xchainer.XchainerError):
            array.cleargrad(graph_1)

        array.require_grad(graph_1).set_grad(grad, graph_1)
        assert array.get_grad(graph_1) is not None
        assert array.get_grad(graph_1)._debug_flat_data == grad._debug_flat_data

        array.cleargrad(graph_1)  # clear
        assert array.get_grad(graph_1) is None

    # keyword arguments
    with xchainer.graph_scope('graph_2') as graph_2:
        with pytest.raises(xchainer.XchainerError):
            array.get_grad(graph_id=graph_2)
        with pytest.raises(xchainer.XchainerError):
            array.set_grad(grad, graph_id=graph_2)
        with pytest.raises(xchainer.XchainerError):
            array.cleargrad(graph_id=graph_2)

        array.require_grad(graph_id=graph_2).set_grad(grad, graph_id=graph_2)
        assert array.get_grad(graph_2) is not None
        assert array.get_grad(graph_id=graph_2) is not None
        assert array.get_grad(graph_2)._debug_flat_data == grad._debug_flat_data
        assert array.get_grad(graph_id=graph_2)._debug_flat_data == grad._debug_flat_data

        array.cleargrad(graph_id=graph_2)  # clear
        assert array.get_grad(graph_2) is None
        assert array.get_grad(graph_id=graph_2) is None


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
    array = xchainer.ndarray(shape, xchainer.float32, [1., 1., 1.])
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

    with xchainer.graph_scope('graph_1') as graph_1, \
            xchainer.graph_scope('graph_2') as graph_2, \
            xchainer.graph_scope('graph_3') as graph_3:

        x1.require_grad(graph_1)
        x2.require_grad(graph_2)

        assert x1.is_grad_required(graph_1)
        assert x2.is_grad_required(graph_2)

        assert not x1.is_grad_required(graph_2)
        assert not x2.is_grad_required(graph_1)

        y = x1 * x2

        assert y.is_grad_required(graph_1)
        assert y.is_grad_required(graph_2)

        # No unspecified graphs are generated
        assert not y.is_grad_required(None)
        assert not y.is_grad_required(graph_3)


@pytest.mark.parametrize('expected_error,invalid_shape,invalid_dtype,invalid_device', [
    (xchainer.DtypeError, None, xchainer.int8, None),
    (xchainer.DimensionError, (2, 1), None, None),
    (xchainer.DeviceError, None, None, 'native:1'),
])
def test_array_grad_invalid_grad(expected_error, invalid_shape, invalid_dtype, invalid_device):
    shape = (3, 1)
    dtype = xchainer.float32
    device = 'native:0'

    array = xchainer.ndarray(shape, dtype, [1., 1., 1.], device=device)
    array.require_grad()

    grad_shape = shape if invalid_shape is None else invalid_shape
    grad_dtype = dtype if invalid_dtype is None else invalid_dtype
    grad_data = [1] * array_utils.total_size(grad_shape)
    grad_device = device if invalid_device is None else invalid_device
    invalid_grad = xchainer.ndarray(grad_shape, grad_dtype, grad_data, grad_device)

    with pytest.raises(expected_error):
        array.set_grad(invalid_grad)
    with pytest.raises(expected_error):
        array.grad = invalid_grad


def test_array_backward():
    with xchainer.graph_scope('graph_1') as graph_1:
        x1 = xchainer.ndarray((3, 1), xchainer.int8, [1, 1, 1]).require_grad(graph_id=graph_1)
        x2 = xchainer.ndarray((3, 1), xchainer.int8, [1, 1, 1]).require_grad(graph_id=graph_1)
        y = x1 * x2

        y.backward(graph_id=graph_1, enable_double_backprop=True)
        gx1 = x1.get_grad(graph_id=graph_1)
        x1.set_grad(None, graph_id=graph_1)

        gx1.backward(graph_id=graph_1)
        assert gx1.get_grad(graph_id=graph_1) is not None


@xchainer.testing.numpy_xchainer_array_equal(strides_check=False)
@pytest.mark.parametrize('value', [-1, 0, 1, 2, 2.3, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_fill(xp, shape, dtype, value, device):
    a = xp.empty(shape, dtype)
    a.fill(value)
    return a


@xchainer.testing.numpy_xchainer_array_equal(strides_check=False)
@pytest.mark.parametrize('value', [-1, 0, 1, 2, 2.3, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_fill_with_scalar(xp, device, shape, dtype, value):
    a = xp.empty(shape, dtype)
    if xp is xchainer:
        value = xchainer.Scalar(value, dtype)
    a.fill(value)
    return a


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('slice1', [(0, 30, 1), (30, 0, -1), (10, 40, 7), (40, 10, -7)])
@pytest.mark.parametrize('slice2', [(0, 50, 1), (50, 0, -1), (10, 40, 7), (40, 10, -7)])
def test_array_tonumpy_identity(device, slice1, slice2):
    start1, end1, step1 = slice1
    start2, end2, step2 = slice2
    x = numpy.arange(1500).reshape((30, 50))[start1:end1:step1, start2:end2:step2]
    y = xchainer.array(x)
    z = xchainer.tonumpy(y)
    xchainer.testing.assert_array_equal_ex(x, y, strides_check=False)
    xchainer.testing.assert_array_equal_ex(x, z, strides_check=False)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('slice1', [(0, 30, 1), (30, 0, -1), (10, 40, 7), (40, 10, -7)])
@pytest.mark.parametrize('slice2', [(0, 50, 1), (50, 0, -1), (10, 40, 7), (40, 10, -7)])
def test_asarray_tonumpy_identity(device, slice1, slice2):
    start1, end1, step1 = slice1
    start2, end2, step2 = slice2
    x = numpy.arange(1500).reshape((30, 50))[start1:end1:step1, start2:end2:step2]
    y = xchainer.asarray(x)
    z = xchainer.tonumpy(y)
    xchainer.testing.assert_array_equal_ex(x, y)
    xchainer.testing.assert_array_equal_ex(x, z, strides_check=False)
