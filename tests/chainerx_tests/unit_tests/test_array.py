import copy
import math
import pickle

import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils


def _check_array(
        array, expected_dtype, expected_shape, expected_data_list=None,
        device=None):
    expected_dtype = chainerx.dtype(expected_dtype)

    assert isinstance(array.dtype, chainerx.dtype)
    assert isinstance(array.shape, tuple)
    assert array.dtype == expected_dtype
    assert array.shape == expected_shape
    assert array.itemsize == expected_dtype.itemsize
    assert array.size == array_utils.total_size(expected_shape)
    assert array.nbytes == expected_dtype.itemsize * \
        array_utils.total_size(expected_shape)
    if expected_data_list is not None:
        assert array._debug_flat_data == expected_data_list

    assert array.is_contiguous

    array_utils.check_device(array, device)


@chainerx.testing.parametrize_dtype_specifier('dtype_spec')
def test_init(shape, dtype_spec):
    array = chainerx.ndarray(shape, dtype_spec)
    _check_array(array, dtype_spec, shape)


@pytest.mark.parametrize(
    'device', [None, 'native:1', chainerx.get_device('native:1')])
@chainerx.testing.parametrize_dtype_specifier('dtype_spec')
def test_init_with_device(shape, dtype_spec, device):
    array = chainerx.ndarray(shape, dtype_spec, device=device)
    _check_array(array, dtype_spec, shape, device=device)


@pytest.mark.parametrize('value', [
    0, 1, -1, 0.1, 0.9, -0.1, -0.9, 1.1, -1.1, 1.9, -
    1.9, True, False, float('inf'), -float('inf'), float('nan'), -0.0
])
@pytest.mark.parametrize('shape', [
    (), (1,), (1, 1, 1)
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_cast_scalar(device, value, shape, dtype):
    np_dtype = numpy.dtype(dtype)
    try:
        np_value = np_dtype.type(value)
    except (ValueError, OverflowError):
        return

    a_np = numpy.asarray([np_value], dtype).reshape(shape)
    a_chx = chainerx.array(a_np)

    def should_cast_succeed(typ):
        try:
            typ(np_value)
            return True
        except (ValueError, OverflowError):
            return False

    # Cast to float
    if should_cast_succeed(float):
        assert type(float(a_chx)) is float
        if math.isnan(float(a_np)):
            assert math.isnan(float(a_chx))
        else:
            assert float(a_np) == float(a_chx)
    # Cast to int
    if should_cast_succeed(int):
        assert type(int(a_chx)) is int
        assert int(a_np) == int(a_chx)
    # Cast to bool
    if should_cast_succeed(bool):
        assert type(bool(a_chx)) is bool
        assert bool(a_np) == bool(a_chx)

    # item()
    item_actual = a_chx.item()
    np_dtype = numpy.dtype(dtype)
    item_expected = np_dtype.type(value).item()
    assert isinstance(item_actual, type(item_expected))
    assert (
        (numpy.isnan(item_actual) and numpy.isnan(item_expected))
        or item_actual == item_expected)


@pytest.mark.parametrize('shape', [
    (0,), (1, 0), (2,), (1, 2), (2, 3),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_cast_scalar_invalid(device, shape):
    dtype = chainerx.float32

    a = chainerx.ones(shape, dtype)
    with pytest.raises(chainerx.DimensionError):
        float(a)

    a = chainerx.ones(shape, dtype)
    with pytest.raises(chainerx.DimensionError):
        int(a)

    a = chainerx.ones(shape, dtype)
    with pytest.raises(chainerx.DimensionError):
        bool(a)

    a = chainerx.ones(shape, dtype)
    with pytest.raises(chainerx.DimensionError):
        a.item()


def test_to_device():
    a = chainerx.ones((2,), chainerx.float32, device='native:0')
    dst_device = chainerx.get_device('native:1')

    b0 = a.to_device(dst_device)  # by device instance
    assert b0.device is dst_device
    chainerx.testing.assert_array_equal_ex(a, b0)

    b1 = a.to_device('native:1')  # by device name
    assert b1.device is dst_device
    chainerx.testing.assert_array_equal_ex(a, b1)

    b2 = a.to_device('native', 1)  # by backend name and index
    assert b2.device is dst_device
    chainerx.testing.assert_array_equal_ex(a, b2)


def _check_to_numpy(a_np, a_chx, device, copy):
    chainerx.testing.assert_array_equal_ex(a_chx, a_np, strides_check=False)
    if a_np.size > 0:
        # test buffer is shared or not
        a_np.fill(1)
        expected = not copy and device.backend.name == 'native'
        actual = numpy.array_equal(a_np, chainerx.to_numpy(a_chx))
        assert expected == actual


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('copy', [True, False])
def test_to_numpy(shape, dtype, device, copy):
    a_chx = array_utils.create_dummy_ndarray(chainerx, shape, dtype)
    a_np = chainerx.to_numpy(a_chx, copy)
    _check_to_numpy(a_np, a_chx, device, copy)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('copy', [True, False])
def test_to_numpy_non_contiguous(shape, dtype, device, copy):
    a_chx = array_utils.create_dummy_ndarray(chainerx, shape, dtype).T
    a_np = chainerx.to_numpy(a_chx, copy)
    _check_to_numpy(a_np, a_chx, device, copy)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('copy', [True, False])
def test_to_numpy_positive_offset(device, copy):
    a_chx = chainerx.arange(6).reshape(2, 3)[:, 1:]
    a_np = chainerx.to_numpy(a_chx, copy)
    _check_to_numpy(a_np, a_chx, device, copy)


def test_view(shape, dtype):
    array = array_utils.create_dummy_ndarray(chainerx, shape, dtype)
    view = array.view()

    chainerx.testing.assert_array_equal_ex(view, array)
    assert view.device is chainerx.get_default_device()

    # inplace modification
    if array.size > 0:
        array += array
        assert array._debug_flat_data == view._debug_flat_data


def test_view_must_not_share_properties():
    array = chainerx.array([3.0], chainerx.float32)
    view = array.view()
    # Test preconditions
    assert not array.is_grad_required()
    assert not view.is_grad_required()
    assert not array.is_backprop_required()
    assert not view.is_backprop_required()

    array.require_grad()
    assert array.is_grad_required()
    assert array.is_backprop_required()
    assert not view.is_grad_required(
    ), 'A view must not share is_grad_required with the original array.'
    assert not view.is_backprop_required(
    ), 'A view must not share is_backprop_required with the original array.'


@chainerx.testing.numpy_chainerx_array_equal(strides_check=False)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('copy', [False, True])
# TODO(beam2d): use fixtures.
@pytest.mark.parametrize(
    'src_dtype',
    ['bool_', 'uint8', 'int8', 'int16', 'int32', 'int64', 'float32',
     'float64'])
@pytest.mark.parametrize(
    'dst_dtype',
    ['bool_', 'uint8', 'int8', 'int16', 'int32', 'int64', 'float32',
     'float64'])
def test_astype(xp, shape, device, copy, src_dtype, dst_dtype):
    a = array_utils.create_dummy_ndarray(xp, shape, src_dtype)

    # Casting negative value to unsigned int behaves different in CUDA
    if device.name == 'cuda:0' and \
            src_dtype in chainerx.testing.signed_dtypes and \
            dst_dtype in chainerx.testing.unsigned_dtypes:
        a = xp.maximum(a, 0)

    b = a.astype(dst_dtype, copy=copy)
    assert a is b if src_dtype == dst_dtype and not copy else a is not b
    return b


def test_as_grad_stopped_copy(shape, float_dtype):
    dtype = float_dtype

    def check(array_a, array_b):
        chainerx.testing.assert_array_equal_ex(
            array_a, array_b, strides_check=False)

        assert array_b.is_contiguous

        # Check memory addresses only if >0 bytes are allocated
        if array_a.size > 0:
            assert (array_a._debug_data_memory_address
                    != array_b._debug_data_memory_address)

    # Stop gradients on all graphs
    with chainerx.backprop_scope('bp1') as bp1, \
            chainerx.backprop_scope('bp2') as bp2, \
            chainerx.backprop_scope('bp3') as bp3:

        a = array_utils.create_dummy_ndarray(chainerx, shape, dtype)
        a.require_grad(bp1)
        a.require_grad(bp2)
        assert a.is_grad_required(bp1)
        assert a.is_grad_required(bp2)
        assert a.is_backprop_required(bp1)
        assert a.is_backprop_required(bp2)
        b = a.as_grad_stopped(copy=True)

        check(a, b)
        assert not b.is_grad_required(bp1)
        assert not b.is_grad_required(bp2)
        assert not b.is_backprop_required(bp1)
        assert not b.is_backprop_required(bp2)

        assert a.is_grad_required(bp1)
        assert a.is_grad_required(bp2)
        assert a.is_backprop_required(bp1)
        assert a.is_backprop_required(bp2)

        # Stop gradients on some graphs
        a = array_utils.create_dummy_ndarray(chainerx, shape, dtype)
        a.require_grad(bp1)
        a.require_grad(bp2)
        a.require_grad(bp3)
        assert a.is_grad_required(bp1)
        assert a.is_grad_required(bp2)
        assert a.is_grad_required(bp3)
        assert a.is_backprop_required(bp1)
        assert a.is_backprop_required(bp2)
        assert a.is_backprop_required(bp3)
        b = a.as_grad_stopped([bp1, bp2], copy=True)

        check(a, b)
        assert not b.is_grad_required(bp1)
        assert not b.is_grad_required(bp2)
        assert not b.is_grad_required(bp3)
        assert not b.is_backprop_required(bp1)
        assert not b.is_backprop_required(bp2)
        assert b.is_backprop_required(bp3)

        assert a.is_grad_required(bp1)
        assert a.is_grad_required(bp2)
        assert a.is_grad_required(bp3)
        assert a.is_backprop_required(bp1)
        assert a.is_backprop_required(bp2)
        assert a.is_backprop_required(bp3)


def test_as_grad_stopped_view(shape, float_dtype):
    dtype = float_dtype

    # Stop gradients on all graphs
    with chainerx.backprop_scope('bp1') as bp1, \
            chainerx.backprop_scope('bp2') as bp2, \
            chainerx.backprop_scope('bp3') as bp3:

        a = array_utils.create_dummy_ndarray(chainerx, shape, dtype)
        a.require_grad(bp1)
        a.require_grad(bp2)
        assert a.is_grad_required(bp1)
        assert a.is_grad_required(bp2)
        assert a.is_backprop_required(bp1)
        assert a.is_backprop_required(bp2)
        b = a.as_grad_stopped(copy=False)

        chainerx.testing.assert_array_equal_ex(a, b)
        assert b.device is a.device
        assert not b.is_grad_required(bp1)
        assert not b.is_grad_required(bp2)
        assert not b.is_backprop_required(bp1)
        assert not b.is_backprop_required(bp2)

        assert a.is_backprop_required(bp1)
        assert a.is_backprop_required(bp2)

        # Stop gradients on some graphs
        a = array_utils.create_dummy_ndarray(chainerx, shape, dtype)
        a.require_grad(bp1)
        a.require_grad(bp2)
        a.require_grad(bp3)
        assert a.is_grad_required(bp1)
        assert a.is_grad_required(bp2)
        assert a.is_grad_required(bp3)
        assert a.is_backprop_required(bp1)
        assert a.is_backprop_required(bp2)
        assert a.is_backprop_required(bp3)
        b = a.as_grad_stopped([bp1, bp2], copy=False)

        chainerx.testing.assert_array_equal_ex(a, b)
        assert b.device is a.device
        assert not b.is_grad_required(bp1)
        assert not b.is_grad_required(bp2)
        assert not b.is_grad_required(bp3)
        assert not b.is_backprop_required(bp1)
        assert not b.is_backprop_required(bp2)
        assert b.is_backprop_required(bp3)

        assert a.is_grad_required(bp1)
        assert a.is_grad_required(bp2)
        assert a.is_grad_required(bp3)
        assert a.is_backprop_required(bp1)
        assert a.is_backprop_required(bp2)
        assert a.is_backprop_required(bp3)


def test_array_repr():
    array = chainerx.array([], chainerx.bool_)
    assert ('array([], shape=(0,), dtype=bool, '
            'device=\'native:0\')' == str(array))

    array = chainerx.array([False], chainerx.bool_)
    assert ('array([False], shape=(1,), dtype=bool, '
            'device=\'native:0\')' == str(array))

    array = chainerx.array([[0, 1, 2], [3, 4, 5]], chainerx.int8)
    assert ('array([[0, 1, 2],\n'
            '       [3, 4, 5]], shape=(2, 3), dtype=int8, '
            'device=\'native:0\')') == str(array)

    array = chainerx.array([[0, 1, 2], [3.25, 4, 5]], chainerx.float32)
    assert ('array([[0.  , 1.  , 2.  ],\n'
            '       [3.25, 4.  , 5.  ]], shape=(2, 3), dtype=float32, '
            'device=\'native:0\')') == str(array)


def test_array_repr_default_backprop_id():
    array = chainerx.array([3.0], chainerx.float32)
    array.require_grad()
    assert ('array([3.], shape=(1,), dtype=float32, device=\'native:0\', '
            'backprop_ids=[\'<default>\'])' == str(array))


def test_array_repr_expired_backprop_id():
    with chainerx.backprop_scope('bp1') as bp1:
        array = chainerx.array([3.0], chainerx.float32)
        array.require_grad(bp1)
    assert ('array([3.], shape=(1,), dtype=float32, device=\'native:0\', '
            'backprop_ids=[\'<expired>\'])' == str(array))


@pytest.mark.parametrize('backprop_args', [(None,), ()])
def test_array_require_grad_without_backprop_id(backprop_args):
    array = chainerx.array([1, 1, 1], chainerx.float32)

    assert not array.is_grad_required(*backprop_args)
    assert not array.is_backprop_required(*backprop_args)
    assert not array.is_backprop_required(chainerx.anygraph)
    array.require_grad(*backprop_args)
    assert array.is_grad_required(*backprop_args)
    assert array.is_backprop_required(*backprop_args)
    assert array.is_backprop_required(chainerx.anygraph)

    # Repeated calls should not fail, but do nothing
    array.require_grad(*backprop_args)
    assert array.is_grad_required(*backprop_args)
    assert array.is_backprop_required(*backprop_args)
    assert array.is_backprop_required(chainerx.anygraph)


def test_array_require_grad_with_backprop_id():
    array = chainerx.array([1, 1, 1], chainerx.float32)

    with chainerx.backprop_scope('bp1') as bp1:
        assert not array.is_backprop_required(bp1)
        array.require_grad(bp1)
        assert array.is_grad_required(bp1)
        assert array.is_backprop_required(bp1)

        # Repeated calls should not fail, but do nothing
        array.require_grad(bp1)
        assert array.is_grad_required(bp1)
        assert array.is_backprop_required(bp1)

    # keyword arguments
    with chainerx.backprop_scope('bp2') as bp2:
        assert not array.is_backprop_required(backprop_id=bp2)
        array.require_grad(backprop_id=bp2)
        assert array.is_grad_required(bp2)
        assert array.is_grad_required(backprop_id=bp2)
        assert array.is_backprop_required(bp2)
        assert array.is_backprop_required(backprop_id=bp2)

        # Repeated calls should not fail, but do nothing
        array.require_grad(backprop_id=bp2)
        assert array.is_grad_required(backprop_id=bp2)
        assert array.is_backprop_required(backprop_id=bp2)


@pytest.mark.parametrize('backprop_args', [(None,), ()])
def test_array_grad_without_backprop_id(backprop_args):
    array = chainerx.array([1., 1., 1.], chainerx.float32)
    grad = chainerx.array([0.5, 0.5, 0.5], chainerx.float32)

    with pytest.raises(chainerx.ChainerxError):
        array.get_grad(*backprop_args)
    with pytest.raises(chainerx.ChainerxError):
        array.set_grad(grad, *backprop_args)
    with pytest.raises(chainerx.ChainerxError):
        array.cleargrad(*backprop_args)

    # Gradient methods
    array.require_grad().set_grad(grad, *backprop_args)
    assert array.get_grad(*backprop_args) is not None
    assert array.get_grad(
        *backprop_args)._debug_flat_data == grad._debug_flat_data

    array.cleargrad(*backprop_args)  # clear
    assert array.get_grad(*backprop_args) is None

    array.set_grad(grad, *backprop_args)
    assert array.get_grad(*backprop_args) is not None
    assert array.get_grad(
        *backprop_args)._debug_flat_data == grad._debug_flat_data

    array.set_grad(None, *backprop_args)  # clear
    assert array.get_grad(*backprop_args) is None

    # Gradient attributes
    array.grad = grad
    assert array.get_grad(*backprop_args) is not None
    assert array.get_grad(*backprop_args) is array.grad

    array.grad = None  # clear
    assert array.get_grad(*backprop_args) is None


def test_array_grad_with_backprop_id():
    array = chainerx.array([1., 1., 1.], chainerx.float32)
    grad = chainerx.array([0.5, 0.5, 0.5], chainerx.float32)

    with chainerx.backprop_scope('bp1') as bp1:
        with pytest.raises(chainerx.ChainerxError):
            array.get_grad(bp1)
        with pytest.raises(chainerx.ChainerxError):
            array.set_grad(grad, bp1)
        with pytest.raises(chainerx.ChainerxError):
            array.cleargrad(bp1)

        array.require_grad(bp1).set_grad(grad, bp1)
        assert array.get_grad(bp1) is not None
        assert array.get_grad(bp1)._debug_flat_data == grad._debug_flat_data

        array.cleargrad(bp1)  # clear
        assert array.get_grad(bp1) is None

    # keyword arguments
    with chainerx.backprop_scope('bp2') as bp2:
        with pytest.raises(chainerx.ChainerxError):
            array.get_grad(backprop_id=bp2)
        with pytest.raises(chainerx.ChainerxError):
            array.set_grad(grad, backprop_id=bp2)
        with pytest.raises(chainerx.ChainerxError):
            array.cleargrad(backprop_id=bp2)

        array.require_grad(backprop_id=bp2).set_grad(grad, backprop_id=bp2)
        assert array.get_grad(bp2) is not None
        assert array.get_grad(backprop_id=bp2) is not None
        assert array.get_grad(bp2)._debug_flat_data == grad._debug_flat_data
        assert array.get_grad(
            backprop_id=bp2)._debug_flat_data == grad._debug_flat_data

        array.cleargrad(backprop_id=bp2)  # clear
        assert array.get_grad(bp2) is None
        assert array.get_grad(backprop_id=bp2) is None


def test_array_grad_no_deepcopy():
    dtype = chainerx.float32
    array = chainerx.array([2, 5, 1], dtype)
    grad = chainerx.array([5, 7, 8], dtype)

    # Set grad
    array.require_grad().set_grad(grad)

    # Retrieve grad twice and assert they share the same underlying data
    grad1 = array.get_grad()
    grad2 = array.get_grad()

    grad1 *= chainerx.array([2, 2, 2], dtype)
    assert grad2._debug_flat_data == [
        10, 14, 16], 'grad getter must not incur a copy'


def test_array_cleargrad():
    dtype = chainerx.float32
    array = chainerx.array([2, 5, 1], dtype)
    grad = chainerx.array([5, 7, 8], dtype)

    # Set grad, get it and save it
    array.require_grad().set_grad(grad)
    del grad
    saved_grad = array.get_grad()

    # Clear grad
    array.cleargrad()
    assert array.get_grad() is None

    assert saved_grad._debug_flat_data == [
        5, 7, 8], 'Clearing grad must not affect previously retrieved grad'


def test_array_grad_identity():
    array = chainerx.array([1., 1., 1.], chainerx.float32)
    grad = chainerx.array([0.5, 0.5, 0.5], chainerx.float32)
    array.require_grad().set_grad(grad)

    assert array.get_grad() is grad, (
        'grad must preserve physical identity')
    assert array.get_grad() is grad, (
        'grad must preserve physical identity in repeated retrieval')

    # array.grad and grad share the same data
    grad += chainerx.array([2, 2, 2], chainerx.float32)
    assert array.get_grad()._debug_flat_data == [
        2.5, 2.5, 2.5], 'A modification to grad must affect array.grad'

    array_grad = array.get_grad()
    array_grad += chainerx.array([1, 1, 1], chainerx.float32)
    assert grad._debug_flat_data == [
        3.5, 3.5, 3.5], 'A modification to array.grad must affect grad'


def test_array_require_grad_multiple_graphs_forward():
    x1 = chainerx.array([1, 1, 1], chainerx.float32)
    x2 = chainerx.array([1, 1, 1], chainerx.float32)

    with chainerx.backprop_scope('bp1') as bp1, \
            chainerx.backprop_scope('bp2') as bp2, \
            chainerx.backprop_scope('bp3') as bp3:

        x1.require_grad(bp1)
        x2.require_grad(bp2)

        assert x1.is_grad_required(bp1)
        assert x2.is_grad_required(bp2)
        assert x1.is_backprop_required(bp1)
        assert x2.is_backprop_required(bp2)

        assert not x1.is_grad_required(bp2)
        assert not x2.is_grad_required(bp1)
        assert not x1.is_backprop_required(bp2)
        assert not x2.is_backprop_required(bp1)

        y = x1 * x2

        assert not y.is_grad_required(bp1)
        assert not y.is_grad_required(bp2)
        assert y.is_backprop_required(bp1)
        assert y.is_backprop_required(bp2)

        # No unspecified graphs are generated
        assert not y.is_backprop_required(None)
        assert not y.is_backprop_required(bp3)


@pytest.mark.parametrize(
    'invalid_shape,invalid_dtype,invalid_device',
    [
        (None, chainerx.float32, None),
        ((2, 1), None, None),
        (None, None, 'native:1'),
    ])
def test_array_grad_invalid_grad(invalid_shape, invalid_dtype, invalid_device):
    shape = (3, 1)
    dtype = chainerx.float64
    device = 'native:0'

    array = chainerx.ones(shape, dtype, device=device)
    array.require_grad()

    grad_shape = shape if invalid_shape is None else invalid_shape
    grad_dtype = dtype if invalid_dtype is None else invalid_dtype
    grad_device = device if invalid_device is None else invalid_device
    invalid_grad = chainerx.ones(
        grad_shape, grad_dtype, device=grad_device)

    with pytest.raises(chainerx.GradientError):
        array.set_grad(invalid_grad)
    with pytest.raises(chainerx.GradientError):
        array.grad = invalid_grad


def test_array_backward():
    with chainerx.backprop_scope('bp1') as bp1:
        x1 = chainerx.array(
            [1, 1, 1], chainerx.float32).require_grad(backprop_id=bp1)
        x2 = chainerx.array(
            [1, 1, 1], chainerx.float32).require_grad(backprop_id=bp1)
        y = x1 * x2

        y.backward(backprop_id=bp1, enable_double_backprop=True)
        gx1 = x1.get_grad(backprop_id=bp1)
        x1.set_grad(None, backprop_id=bp1)

        gx1.backward(backprop_id=bp1)
        with pytest.raises(chainerx.ChainerxError):
            gx1.get_grad(backprop_id=bp1)


@chainerx.testing.numpy_chainerx_array_equal(strides_check=False)
@pytest.mark.parametrize(
    'value', [-1, 0, 1, 2, 2.3, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_fill(xp, shape, dtype, value, device):
    a = xp.empty(shape, dtype)
    a.fill(value)
    return a


@chainerx.testing.numpy_chainerx_array_equal(strides_check=False)
@pytest.mark.parametrize(
    'value', [-1, 0, 1, 2, 2.3, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_fill_with_scalar(xp, device, shape, dtype, value):
    a = xp.empty(shape, dtype)
    if xp is chainerx:
        value = chainerx.Scalar(value)
    a.fill(value)
    return a


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize(
    'slice1', [(0, 30, 1), (30, 0, -1), (10, 40, 7), (40, 10, -7)])
@pytest.mark.parametrize(
    'slice2', [(0, 50, 1), (50, 0, -1), (10, 40, 7), (40, 10, -7)])
def test_array_to_numpy_identity(device, slice1, slice2):
    start1, end1, step1 = slice1
    start2, end2, step2 = slice2
    x = numpy.arange(1500).reshape((30, 50))[
        start1:end1:step1, start2:end2:step2]
    y = chainerx.array(x)
    z = chainerx.to_numpy(y)
    chainerx.testing.assert_array_equal_ex(x, y, strides_check=False)
    chainerx.testing.assert_array_equal_ex(x, z, strides_check=False)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize(
    'slice1', [(0, 30, 1), (30, 0, -1), (10, 40, 7), (40, 10, -7)])
@pytest.mark.parametrize(
    'slice2', [(0, 50, 1), (50, 0, -1), (10, 40, 7), (40, 10, -7)])
def test_asarray_to_numpy_identity(device, slice1, slice2):
    start1, end1, step1 = slice1
    start2, end2, step2 = slice2
    x = numpy.arange(1500).reshape((30, 50))[
        start1:end1:step1, start2:end2:step2]
    y = chainerx.asarray(x)
    z = chainerx.to_numpy(y)
    chainerx.testing.assert_array_equal_ex(x, y)
    chainerx.testing.assert_array_equal_ex(x, z, strides_check=False)


# TODO(niboshi): Add pickle test involving context destruction and re-creation
@pytest.mark.parametrize_device(['native:0', 'native:1', 'cuda:0'])
def test_array_pickle(device):
    arr = chainerx.array([1, 2], chainerx.float32, device=device)
    s = pickle.dumps(arr)
    del arr

    arr2 = pickle.loads(s)
    assert isinstance(arr2, chainerx.ndarray)
    assert arr2.device is device
    assert arr2.dtype == chainerx.float32
    chainerx.testing.assert_array_equal(
        arr2,
        chainerx.array([1, 2], chainerx.float32))


# TODO(niboshi): Add deepcopy test with arbitrary context
@pytest.mark.parametrize_device(['native:0', 'native:1', 'cuda:0'])
def test_array_deepcopy(device):
    arr = chainerx.array([1, 2], chainerx.float32, device=device)
    arr2 = copy.deepcopy(arr)

    assert isinstance(arr2, chainerx.ndarray)
    assert arr2.device is device
    assert arr2.dtype == chainerx.float32
    chainerx.testing.assert_array_equal(
        arr2,
        chainerx.array([1, 2], chainerx.float32))
