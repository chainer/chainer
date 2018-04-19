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


@pytest.fixture(params=[True, False])
def is_module(request):
    return request.param


def _create_dummy_data(shape, dtype, pattern=1):
    assert isinstance(dtype, str)

    total_size = _total_size(shape)
    if pattern == 1:
        if dtype in ('bool', 'bool_'):
            return [i % 2 == 1 for i in range(total_size)]
        else:
            return [i for i in range(total_size)]
    else:
        if dtype in ('bool', 'bool_'):
            return [i % 3 == 0 for i in range(total_size)]
        else:
            return [1 + i for i in range(total_size)]


def _create_dummy_ndarray(shape, dtype):
    assert isinstance(dtype, str)
    return numpy.arange(_total_size(shape)).reshape(shape).astype(dtype)


def _check_array(array, expected_dtype, expected_shape, expected_data_list, expected_is_contiguous=True, device=None):
    assert isinstance(expected_dtype, str)
    expected_dtype = xchainer.dtype(expected_dtype)

    assert isinstance(array.dtype, xchainer.dtype)
    assert isinstance(array.shape, tuple)
    assert array.dtype == expected_dtype
    assert array.shape == expected_shape
    assert array.element_bytes == expected_dtype.itemsize
    assert array.total_size == _total_size(expected_shape)
    assert array.total_bytes == expected_dtype.itemsize * _total_size(expected_shape)
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
    assert array_a.element_bytes == array_b.element_bytes
    assert array_a.total_size == array_b.total_size
    assert array_a.total_bytes == array_b.total_bytes
    assert array_a._debug_flat_data == array_b._debug_flat_data


def _check_arrays_equal_copy(array_a, array_b):
    _check_arrays_equal(array_a, array_b)
    assert array_b.is_contiguous
    assert 0 == array_b.offset

    # Check memory addresses only if >0 bytes are allocated
    if array_a.total_size > 0:
        assert array_a._debug_data_memory_address != array_b._debug_data_memory_address


def _check_array_equals_ndarray(array, ndarray):
    assert array.shape == ndarray.shape
    assert array.total_size == ndarray.size
    assert array.ndim == ndarray.ndim
    assert array.element_bytes == ndarray.itemsize
    assert array.total_bytes == ndarray.itemsize * ndarray.size
    numpy.testing.assert_array_equal(array._debug_flat_data, ndarray.ravel().tolist())
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


def _total_size(shape):
    return functools.reduce(operator.mul, shape, 1)


def _check_init(shape, dtype_spec, device=None, with_device=True):
    expected_dtype = xchainer.dtype(dtype_spec).name
    data_list = _create_dummy_data(shape, expected_dtype)

    if with_device:
        array = xchainer.ndarray(shape, dtype_spec, data_list, device)
    else:
        array = xchainer.ndarray(shape, dtype_spec, data_list)

    _check_array(array, expected_dtype, shape, data_list, device=device)


@xchainer.testing.parametrize_dtype_specifier('dtype_spec')
def test_init_without_device(shape, dtype_spec):
    _check_init(shape, dtype_spec, with_device=False)


@pytest.mark.parametrize('device', [None, 'native:1', xchainer.get_device('native:1')])
@xchainer.testing.parametrize_dtype_specifier('dtype_spec')
def test_init_with_device(shape, dtype_spec, device):
    _check_init(shape, dtype_spec, device=device)


def _check_numpy_init(ndarray, device=None):
    shape = ndarray.shape
    if device is None:
        array = xchainer.ndarray(ndarray)
    else:
        array = xchainer.ndarray(ndarray, device)

    ndarray_is_contigous = ndarray.flags['C_CONTIGUOUS']
    _check_array(
        array, ndarray.dtype.name, shape, ndarray.ravel().tolist(),
        expected_is_contiguous=ndarray_is_contigous, device=device)
    _check_array_equals_ndarray(array, ndarray)

    # test possibly freed memory
    data_copy = ndarray.copy()
    del ndarray
    assert array._debug_flat_data == data_copy.ravel().tolist()

    # recovered data should be equal
    data_recovered = numpy.array(array)
    _check_ndarray_equal_ndarray(data_copy, data_recovered, skip_strides=True, skip_flags=True)

    # recovered data should be a copy
    data_recovered_to_modify = numpy.array(array)
    data_recovered_to_modify *= _create_dummy_ndarray(shape, data_copy.dtype.name)
    _check_array_equals_ndarray(array, data_recovered)


def test_numpy_init(shape, dtype):
    ndarray = _create_dummy_ndarray(shape, dtype)
    _check_numpy_init(ndarray)


def test_numpy_non_contiguous_init(shape, dtype):
    ndarray = _create_dummy_ndarray(shape, dtype)
    _check_numpy_init(ndarray.T)


def test_numpy_init_with_offset():
    ndarray = _create_dummy_ndarray((2, 3), 'int32')
    a = xchainer.array(ndarray)
    numpy.testing.assert_array_equal(numpy.array(a[1:]), ndarray[1:])


def test_numpy_init_device(shape, dtype):
    ndarray = _create_dummy_ndarray(shape, dtype)
    _check_numpy_init(ndarray, xchainer.get_device('native:1'))


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


@pytest.mark.parametrize('value', [
    0, 1, -1, 0.1, 0.9, -0.1, -0.9, 1.1, -1.1, 1.9, -1.9, True, False, float('inf'), -float('inf'), float('nan'), -0.0
])
@pytest.mark.parametrize('shape', [
    (), (1,), (1, 1, 1)
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_asscalar(device, value, shape, dtype):
    np_dtype = numpy.dtype(dtype)
    try:
        np_value = np_dtype.type(value)
    except (ValueError, OverflowError):
        return

    a_np = numpy.asarray([np_value], dtype).reshape(shape)
    a_xc = xchainer.ndarray(a_np)

    def should_cast_succeed(typ):
        try:
            typ(np_value)
            return True
        except (ValueError, OverflowError):
            return False

    # Cast to float
    if should_cast_succeed(float):
        assert type(float(a_xc)) is float
        if math.isnan(float(a_np)):
            assert math.isnan(float(a_xc))
        else:
            assert float(a_np) == float(a_xc)
    # Cast to int
    if should_cast_succeed(int):
        assert type(int(a_xc)) is int
        assert int(a_np) == int(a_xc)
    # Cast to bool
    if should_cast_succeed(bool):
        assert type(bool(a_xc)) is bool
        assert bool(a_np) == bool(a_xc)

    # xchainer.asscalar
    assert isinstance(xchainer.asscalar(a_xc), type(numpy.asscalar(a_np)))
    if math.isnan(numpy.asscalar(a_np)):
        assert math.isnan(xchainer.asscalar(a_xc))
    else:
        assert xchainer.asscalar(a_xc) == numpy.asscalar(a_np)


@pytest.mark.parametrize('shape', [
    (0,), (1, 0), (2,), (1, 2), (2, 3),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_invalid_asscalar(device, shape):
    dtype = xchainer.float32

    a = xchainer.ones(shape, dtype)
    with pytest.raises(xchainer.DimensionError):
        xchainer.asscalar(a)

    a = xchainer.ones(shape, dtype)
    with pytest.raises(xchainer.DimensionError):
        float(a)

    a = xchainer.ones(shape, dtype)
    with pytest.raises(xchainer.DimensionError):
        int(a)

    a = xchainer.ones(shape, dtype)
    with pytest.raises(xchainer.DimensionError):
        bool(a)


@xchainer.testing.numpy_xchainer_array_equal()
def test_transpose(xp, shape, dtype):
    ndarray = _create_dummy_ndarray(shape, dtype)
    array = xp.array(ndarray)
    return array.transpose()


@xchainer.testing.numpy_xchainer_array_equal()
def test_T(xp, shape, dtype):
    ndarray = _create_dummy_ndarray(shape, dtype)
    array = xp.array(ndarray)
    return array.T


@xchainer.testing.numpy_xchainer_array_equal()
def test_module_transpose(xp, shape, dtype):
    ndarray = _create_dummy_ndarray(shape, dtype)
    array = xp.array(ndarray)
    return xp.transpose(array)


@pytest.mark.parametrize('a_shape,b_shape', [
    ((), ()),
    ((0,), (0,)),
    ((1,), (1,)),
    ((5,), (5,)),
    ((2, 3), (2, 3)),
    ((1,), ()),
    ((), (1,)),
    ((1, 1), ()),
    ((), (1, 1)),
    ((6,), (2, 3)),
    ((2, 3), (6,)),
    ((2, 0, 3), (5, 0, 7)),
    ((5,), (1, 1, 5, 1, 1)),
    ((1, 1, 5, 1, 1), (5,)),
    ((2, 3), (3, 2)),
    ((2, 3, 4), (3, 4, 2)),
])
def test_reshape(a_shape, b_shape):
    size = functools.reduce(operator.mul, a_shape, 1)
    dtype = numpy.float32
    a_np = numpy.arange(size, dtype=dtype).reshape(a_shape)
    b_np = a_np.reshape(b_shape)
    a_xc = xchainer.ndarray(a_np)

    def check(b_xc):
        assert b_xc is not a_xc
        assert b_np.shape == b_xc.shape
        assert b_xc.is_contiguous
        assert a_xc._debug_data_memory_address == b_xc._debug_data_memory_address, 'Reshape must be done without copy'
        assert b_xc.strides == b_np.strides, 'Strides after reshape must match NumPy behavior'
        _check_arrays_equal(xchainer.ndarray(b_np), b_xc)

    # instance methods
    check(a_xc.reshape(b_shape))  # by tuple
    check(a_xc.reshape(list(b_shape)))  # by list
    check(a_xc.reshape(*b_shape))  # by variable length args

    # module functions
    check(xchainer.reshape(a_xc, b_shape))  # by tuple
    check(xchainer.reshape(a_xc, list(b_shape)))  # by list
    with pytest.raises(TypeError):
        xchainer.reshape(a_xc, *b_shape)

# TODO(niboshi): Test with non-contiguous input array that requires copy to reshape
# TODO(niboshi): Test with non-contiguous input array that does not require copy to reshape


@pytest.mark.parametrize('shape1,shape2', [
    ((), (0,)),
    ((), (2,)),
    ((), (1, 2,)),
    ((0,), (1,)),
    ((0,), (1, 1, 1)),
    ((2, 3), (2, 3, 2)),
    ((2, 3, 4), (2, 3, 5)),
])
def test_invalid_reshape(shape1, shape2):
    def check(a_shape, b_shape):
        size = functools.reduce(operator.mul, a_shape, 1)
        dtype = numpy.float32
        a_np = numpy.arange(size, dtype=dtype).reshape(a_shape)
        a_xc = xchainer.ndarray(a_np)

        with pytest.raises(xchainer.DimensionError):
            a_xc.reshape(b_shape)

    check(shape1, shape2)
    check(shape2, shape1)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('shape,axis', [
    ((), None),
    ((0,), None),
    ((1,), None),
    ((1, 1), None),
    ((1, 0, 1), None),
    ((3,), None),
    ((3, 1), None),
    ((1, 3), None),
    ((2, 0, 3), None),
    ((2, 4, 3), None),
    ((2, 1, 3), 1),
    ((2, 1, 3), -2),
    ((1, 2, 1, 3, 1, 1, 4), None),
    ((1, 2, 1, 3, 1, 1, 4), (2, 0, 4)),
    ((1, 2, 1, 3, 1, 1, 4), (-2, 0, 4)),
])
def test_squeeze(is_module, xp, shape, axis):
    ndarray = _create_dummy_ndarray(shape, 'float32')
    a = xp.array(ndarray)
    if is_module:
        return xp.squeeze(a, axis)
    else:
        return a.squeeze(axis)


@xchainer.testing.numpy_xchainer_array_equal(accept_error=(xchainer.DimensionError, ValueError))
@pytest.mark.parametrize('shape,axis', [
    ((2, 1, 3), 0),
    ((2, 1, 3), -1),
    ((2, 1, 3), (1, 2)),
    ((2, 1, 3), (1, -1)),
    ((2, 1, 3), (1, 1)),
])
def test_invalid_squeeze(is_module, xp, shape, axis):
    a = xp.ones(shape, xp.float32)
    if is_module:
        return xp.squeeze(a, axis)
    else:
        return a.squeeze(axis)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('src_shape,dst_shape', [
    ((), ()),
    ((1,), (2,)),
    ((1, 1), (2, 2)),
    ((1, 1), (1, 2)),
])
def test_broadcast_to(xp, src_shape, dst_shape):
    ndarray = _create_dummy_ndarray(src_shape, 'float32')
    a = xp.array(ndarray)
    return xp.broadcast_to(a, dst_shape)


@xchainer.testing.numpy_xchainer_array_equal(accept_error=(xchainer.DimensionError, TypeError))
def test_broadcast_to_auto_prefix(xp):
    ndarray = numpy.arange(2, dtype=numpy.float32)
    a = xp.array(ndarray)
    return xp.broadcast_to(a, (3, 2))


@xchainer.testing.numpy_xchainer_array_equal(accept_error=(xchainer.DimensionError, TypeError))
@pytest.mark.parametrize(('src_shape,dst_shape'), [
    ((3,), (2,)),
    ((3,), (3, 2)),
    ((1, 3), (3, 2)),
])
def test_invalid_broadcast_to(xp, src_shape, dst_shape):
    a = xp.ones(src_shape, xchainer.float32)
    return xp.broadcast_to(a, dst_shape)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_copy(xp, shape, dtype, device):
    ndarray = _create_dummy_ndarray(shape, dtype)
    a = xp.array(ndarray)
    return a.copy()


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_module_copy(xp, shape, dtype, device):
    ndarray = _create_dummy_ndarray(shape, dtype)
    a = xp.array(ndarray)
    return xp.copy(a)


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


@pytest.mark.parametrize('a_object,b_object', [
    ([], []),
    ([0], [0]),
    ([0], [-0]),
    ([0], [1]),
    ([0.2], [0.2]),
    ([0.2], [-0.3]),
    ([True], [True]),
    ([True], [False]),
    ([0, 1, 2], [0, 1, 2]),
    ([1, 1, 2], [0, 1, 2]),
    ([0, 1, 2], [1, 2, 3]),
    ([0., numpy.nan], [0., 1.]),
    ([0., numpy.nan], [0., numpy.nan]),
    ([0., numpy.inf], [0., 1.]),
    ([0., -numpy.inf], [0., 1.]),
    ([numpy.inf, 1.], [numpy.inf, 1.]),
    ([-numpy.inf, 1.], [-numpy.inf, 1.]),
    ([numpy.inf, 1.], [-numpy.inf, 1.]),
    ([numpy.inf, 1.], [-numpy.inf, numpy.nan]),
    ([[0, 1], [2, 3]], [[0, 1], [2, 3]]),
    ([[0, 1], [2, 3]], [[0, 1], [2, -2]]),
    ([[0, 1], [2, 3]], [[1, 2], [3, 4]]),
    # broadcast
    (0, [0]),
    (1, [0]),
    ([], [0]),
    ([0], [[0, 1, 2], [3, 4, 5]]),
    ([[0], [1]], [0, 1, 2]),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_eq(device, a_object, b_object, dtype):
    try:
        a_np = numpy.array(a_object, dtype)
        b_np = numpy.array(b_object, dtype)
    except (ValueError, OverflowError):
        # Skip if creating an ndarray while casting the data to the parameterized dtype fails.
        # E.g. [numpy.inf] to numpy.int32.
        return

    a_xc = xchainer.ndarray(a_np)
    b_xc = xchainer.ndarray(b_np)

    _check_array_equals_ndarray(a_xc == b_xc, a_np == b_np)
    _check_array_equals_ndarray(b_xc == a_xc, b_np == a_np)
    _check_array_equals_ndarray(xchainer.equal(a_xc, b_xc), numpy.equal(a_np, b_np))
    _check_array_equals_ndarray(xchainer.equal(b_xc, a_xc), numpy.equal(b_np, a_np))


@pytest.mark.parametrize('a_shape,b_shape', [
    ((2,), (3,)),
    ((2,), (2, 3)),
    ((1, 2, 3), (1, 2, 3, 4)),
])
def test_invalid_eq(a_shape, b_shape):
    def create_ndarray(shape):
        size = functools.reduce(operator.mul, shape, 1)
        dtype = numpy.float32
        return numpy.arange(size, dtype=dtype).reshape(shape)

    def check(a_xc, b_xc):
        with pytest.raises(xchainer.DimensionError):
            a_xc == b_xc

        with pytest.raises(xchainer.DimensionError):
            xchainer.equal(a_xc, b_xc)

    a_np = create_ndarray(a_shape)
    b_np = create_ndarray(b_shape)

    a_xc = xchainer.ndarray(a_np)
    b_xc = xchainer.ndarray(b_np)

    check(a_xc, b_xc)
    check(b_xc, a_xc)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_neg(xp, device, shape, dtype):
    if dtype == 'bool_':  # Checked in test_invalid_bool_neg
        return xp.array([])
    size = functools.reduce(operator.mul, shape, 1)
    obj = numpy.arange(size).reshape(shape).astype(dtype)
    x = xp.array(obj)
    return -x


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_invalid_bool_neg(device):
    def check(xp, err):
        x = xp.array([True, False], dtype='bool_')
        with pytest.raises(err):
            -x

    check(xchainer, xchainer.DtypeError)
    check(numpy, TypeError)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_negative(xp, device, shape, dtype):
    if dtype == 'bool_':  # Checked in test_invalid_bool_neg
        return xp.array([])
    size = functools.reduce(operator.mul, shape, 1)
    obj = numpy.arange(size).reshape(shape).astype(dtype)
    x = xp.array(obj)
    return xp.negative(x)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_invalid_bool_negative(device):
    def check(xp, err):
        x = xp.array([True, False], dtype='bool_')
        with pytest.raises(err):
            xp.negative(x)

    check(xchainer, xchainer.DtypeError)
    check(numpy, TypeError)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_add_iadd(device, shape, dtype):
    lhs_data_list = _create_dummy_data(shape, dtype, pattern=1)
    rhs_data_list = _create_dummy_data(shape, dtype, pattern=2)

    lhs = xchainer.ndarray(shape, dtype, lhs_data_list)
    rhs = xchainer.ndarray(shape, dtype, rhs_data_list)

    expected_data_list = [x + y for x, y in zip(lhs_data_list, rhs_data_list)]
    if dtype == 'bool_':
        expected_data_list = [x > 0 for x in expected_data_list]  # [0, 2] => [False, True]

    def check(out):
        assert out.dtype == xchainer.dtype(dtype)
        assert out.shape == shape
        assert out._debug_flat_data == expected_data_list
        assert lhs._debug_flat_data == lhs_data_list  # operands must not be altered
        assert rhs._debug_flat_data == rhs_data_list

    check(lhs + rhs)
    check(xchainer.add(lhs, rhs))

    lhs_prev = lhs
    lhs += rhs
    assert lhs is lhs_prev, 'inplace operation must not alter lhs reference'
    assert lhs._debug_flat_data == expected_data_list
    assert rhs._debug_flat_data == rhs_data_list


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_sub_isub(device, shape, dtype):
    if dtype == 'bool_':
        # TODO(niboshi): Compare directly with NumPy
        return  # not supported
    lhs_data_list = _create_dummy_data(shape, dtype, pattern=1)
    rhs_data_list = _create_dummy_data(shape, dtype, pattern=2)

    lhs = xchainer.ndarray(shape, dtype, lhs_data_list)
    rhs = xchainer.ndarray(shape, dtype, rhs_data_list)

    if dtype == 'uint8':
        # TODO(niboshi): Compare directly with NumPy
        expected_data_list = [(0x100 + x - y) & 0xff for x, y in zip(lhs_data_list, rhs_data_list)]
    else:
        expected_data_list = [x - y for x, y in zip(lhs_data_list, rhs_data_list)]

    def check(out):
        assert out.dtype == xchainer.dtype(dtype)
        assert out.shape == shape
        assert out._debug_flat_data == expected_data_list
        assert lhs._debug_flat_data == lhs_data_list  # operands must not be altered
        assert rhs._debug_flat_data == rhs_data_list

    check(lhs - rhs)
    check(xchainer.subtract(lhs, rhs))

    lhs_prev = lhs
    lhs -= rhs
    assert lhs is lhs_prev, 'inplace operation must not alter lhs reference'
    assert lhs._debug_flat_data == expected_data_list
    assert rhs._debug_flat_data == rhs_data_list


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_mul_imul(device, shape, dtype):
    lhs_data_list = _create_dummy_data(shape, dtype, pattern=1)
    rhs_data_list = _create_dummy_data(shape, dtype, pattern=2)

    lhs = xchainer.ndarray(shape, dtype, lhs_data_list)
    rhs = xchainer.ndarray(shape, dtype, rhs_data_list)

    expected_data_list = [x * y for x, y in zip(lhs_data_list, rhs_data_list)]
    if dtype == 'bool_':
        expected_data_list = [x > 0 for x in expected_data_list]  # [0, 1] => [False, True]

    def check(out):
        assert out.dtype == xchainer.dtype(dtype)
        assert out.shape == shape
        assert out._debug_flat_data == expected_data_list
        assert lhs._debug_flat_data == lhs_data_list  # operands must not be altered
        assert rhs._debug_flat_data == rhs_data_list

    check(lhs * rhs)
    check(xchainer.multiply(lhs, rhs))

    lhs_prev = lhs
    lhs *= rhs
    assert lhs is lhs_prev, 'inplace operation must not alter lhs reference'
    assert lhs._debug_flat_data == expected_data_list
    assert rhs._debug_flat_data == rhs_data_list


@pytest.mark.parametrize('scalar', [0, -1, 1, 2])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_mul_scalar(scalar, device, shape, dtype):
    data_list = _create_dummy_data(shape, dtype)

    # Implicit casting in NumPy's multiply depends on the 'casting' argument,
    # which is not yet supported (xChainer always casts).
    # Therefore, we explicitly cast the scalar to the dtype of the ndarray
    # before the multiplication for NumPy.
    scalar_np = numpy.dtype(dtype).type(scalar)
    expected = numpy.array(data_list, dtype=dtype).reshape(shape)
    expected *= scalar_np

    x = xchainer.ndarray(shape, dtype, data_list)
    scalar_xc = xchainer.Scalar(scalar, dtype)
    _check_array_equals_ndarray(x * scalar, expected)
    _check_array_equals_ndarray(x * scalar_xc, expected)
    _check_array_equals_ndarray(scalar * x, expected)
    _check_array_equals_ndarray(scalar_xc * x, expected)
    _check_array_equals_ndarray(xchainer.multiply(x, scalar), expected)
    _check_array_equals_ndarray(xchainer.multiply(x, scalar_xc), expected)
    _check_array_equals_ndarray(xchainer.multiply(scalar, x), expected)
    _check_array_equals_ndarray(xchainer.multiply(scalar_xc, x), expected)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_truediv_itruediv(device, shape, dtype):
    if dtype == 'bool_':
        # TODO(niboshi): Compare directly with NumPy
        return  # not supported
    lhs_data_list = _create_dummy_data(shape, dtype, pattern=1)
    rhs_data_list = _create_dummy_data(shape, dtype, pattern=2)

    lhs = xchainer.ndarray(shape, dtype, lhs_data_list)
    rhs = xchainer.ndarray(shape, dtype, rhs_data_list)

    if dtype in ('int8', 'int16', 'int32', 'int64', 'uint8'):
        # TODO(niboshi): The behavior should be true division, but currently it's not supported.
        # Its temporary behavior for integral division is rounding towards zero.
        expected_data_list = [int(x / y) for x, y in zip(lhs_data_list, rhs_data_list)]
    else:
        expected_data_list = [x / y for x, y in zip(lhs_data_list, rhs_data_list)]

    def check(out):
        assert out.dtype == xchainer.dtype(dtype)
        assert out.shape == shape
        numpy.testing.assert_allclose(out._debug_flat_data, expected_data_list, rtol=1e-3)
        assert lhs._debug_flat_data == lhs_data_list  # operands must not be altered
        assert rhs._debug_flat_data == rhs_data_list

    check(lhs / rhs)
    check(xchainer.divide(lhs, rhs))

    lhs_prev = lhs
    lhs /= rhs
    assert lhs is lhs_prev, 'inplace operation must not alter lhs reference'
    numpy.testing.assert_allclose(lhs._debug_flat_data, expected_data_list, rtol=1e-3)
    assert rhs._debug_flat_data == rhs_data_list


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


@xchainer.testing.numpy_xchainer_array_equal()
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


@xchainer.testing.numpy_xchainer_array_equal(type_check=False, accept_error=(xchainer.DimensionError, numpy.AxisError))
@pytest.mark.parametrize("shape,indices,axis", _take_valid_params + _take_invalid_params)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_take(is_module, xp, shape, indices, axis, device):
    a = xp.arange(_total_size(shape)).reshape(shape)

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

    a = xchainer.arange(_total_size(shape)).reshape(shape).astype(dtype).require_grad()
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
@xchainer.testing.numpy_xchainer_array_equal(type_check=False)
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
@xchainer.testing.numpy_xchainer_array_equal(type_check=False)
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
@pytest.mark.parametrize('value', [-1, 0, 1, 2, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_fill(xp, shape, dtype, value, device):
    a = xp.empty(shape, dtype)
    a.fill(value)
    return a


@pytest.mark.parametrize('value', [-1, 0, 1, 2, float('inf'), float('nan')])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_fill_with_scalar(device, shape, dtype, value):
    a_np = numpy.empty(shape, dtype)
    a_xc = xchainer.empty(shape, dtype)
    a_np.fill(value)
    a_xc.fill(xchainer.Scalar(value, dtype))
    a_xc.device.synchronize()
    numpy.testing.assert_array_equal(a_xc, a_np)


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
@xchainer.testing.numpy_xchainer_allclose(rtol=1e-7, atol=0, type_check=False)
# TODO(hvy): Dtype promotion is not supported yet.
def test_logsumexp(xp, device, a_shape, axis, float_dtype, keepdims):
    a = xp.arange(_total_size(a_shape), dtype=float_dtype).reshape(a_shape)
    if xp is numpy:
        return xp.log(xp.sum(xp.exp(a), axis=axis, keepdims=keepdims))
    return xp.logsumexp(a, axis=axis, keepdims=keepdims)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _invalid_logsumexp_params)
@pytest.mark.parametrize('keepdims', [True, False])
# TODO(hvy): Dtype promotion is not supported yet.
# TODO(hvy): Should not overflow for large numbers, add tests
def test_invalid_logsumexp(device, a_shape, axis, float_dtype, keepdims):
    a = xchainer.arange(_total_size(a_shape), dtype=float_dtype).reshape(a_shape)
    with pytest.raises(xchainer.DimensionError):
        xchainer.logsumexp(a, axis=axis, keepdims=keepdims)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _logsumexp_params)
@xchainer.testing.numpy_xchainer_allclose(rtol=1e-7, atol=1e-5, type_check=False)
# TODO(hvy): Dtype promotion is not supported yet.
def test_log_softmax(xp, device, a_shape, axis, float_dtype):
    a = xp.arange(_total_size(a_shape), dtype=float_dtype).reshape(a_shape)
    if xp is numpy:
        # Default is the second axis
        axis = axis if axis is not None else 1
        return a - xp.log(xp.sum(xp.exp(a), axis=axis, keepdims=True))
    return xp.log_softmax(a, axis=axis)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _invalid_logsumexp_params)
# TODO(hvy): Dtype promotion is not supported yet.
def test_invalid_log_softmax(device, a_shape, axis, float_dtype):
    a = xchainer.arange(_total_size(a_shape), dtype=float_dtype).reshape(a_shape)
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
@xchainer.testing.numpy_xchainer_array_equal(accept_error=(ValueError, xchainer.DimensionError))
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
