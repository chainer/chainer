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


def _check_array(array, expected_dtype, expected_shape, expected_total_size, expected_data_list, expected_is_contiguous=True,
                 device_id=None):
    assert isinstance(array.dtype, xchainer.Dtype)
    assert isinstance(array.shape, tuple)
    assert array.dtype == expected_dtype
    assert array.shape == expected_shape
    assert array.element_bytes == expected_dtype.itemsize
    assert array.total_size == expected_total_size
    assert array.total_bytes == expected_dtype.itemsize * expected_total_size
    assert array._debug_flat_data == expected_data_list
    assert array.is_contiguous == expected_is_contiguous
    assert array.offset == 0
    if device_id is None:
        device = xchainer.get_default_device()
    else:
        device = xchainer.get_default_context().get_device(device_id)
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
    assert array._debug_flat_data == ndarray.ravel().tolist()
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


def _check_init(shape, dtype, device=None, with_device=True):
    data_list = _create_dummy_data(shape, dtype)

    if with_device:
        array = xchainer.Array(shape, dtype, data_list, device)
    else:
        array = xchainer.Array(shape, dtype, data_list)

    _check_array(array, dtype, shape, _size(shape), data_list, device_id=device)


def test_init_without_device(array_init_inputs):
    _check_init(*array_init_inputs, with_device=False)


def test_init_with_device(array_init_inputs):
    _check_init(*array_init_inputs, device='native:1')


def test_init_with_none_device(array_init_inputs):
    _check_init(*array_init_inputs, device=None)


def _check_numpy_init(ndarray, shape, dtype, device=None):
    if device is None:
        array = xchainer.Array(ndarray)
    else:
        array = xchainer.Array(ndarray, device)

    ndarray_is_contigous = ndarray.flags['C_CONTIGUOUS']
    _check_array(
        array, dtype, shape, _size(shape), ndarray.ravel().tolist(),
        expected_is_contiguous=ndarray_is_contigous, device_id=device)
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
    data_recovered_to_modify *= _create_dummy_ndarray(shape, data_copy.dtype)
    _check_array_equals_ndarray(array, data_recovered)


def test_numpy_init(array_init_inputs):
    shape, dtype = array_init_inputs
    ndarray = _create_dummy_ndarray(shape, getattr(numpy, dtype.name))
    _check_numpy_init(ndarray, shape, dtype)


def test_numpy_non_contiguous_init(array_init_inputs):
    shape, dtype = array_init_inputs
    ndarray = _create_dummy_ndarray(shape, getattr(numpy, dtype.name))
    _check_numpy_init(ndarray.T, shape[::-1], dtype)


def test_numpy_init_device(array_init_inputs):
    shape, dtype = array_init_inputs
    ndarray = _create_dummy_ndarray(shape, getattr(numpy, dtype.name))
    _check_numpy_init(ndarray, shape, dtype, 'native:1')


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


def test_view(array_init_inputs):
    shape, dtype_name = array_init_inputs
    dtype = xchainer.Dtype(dtype_name)
    data_list = _create_dummy_data(shape, dtype, pattern=1)

    array = xchainer.Array(shape, dtype, data_list)
    view = array.view()

    _check_array(view, dtype, shape, _size(shape), data_list)

    # inplace modification
    if len(data_list) > 0:
        array += array
        assert array._debug_flat_data == view._debug_flat_data


def test_view_must_not_share_properties():
    array = xchainer.Array((1,), xchainer.float32, [3.0])
    view = array.view()
    # Test preconditions
    assert not array.is_grad_required()
    assert not view.is_grad_required()

    array.require_grad()
    assert not view.is_grad_required(), 'A view must not share is_grad_required with the original array.'


def test_transpose(array_init_inputs):
    shape, dtype = array_init_inputs
    data_list = _create_dummy_data(shape, dtype)

    array = xchainer.Array(shape, dtype, data_list)

    def _check_transpose(array_transpose):
        assert shape[::-1] == array_transpose.shape
        assert array.dtype == array_transpose.dtype
        assert array.element_bytes == array_transpose.element_bytes
        assert array.total_size == array_transpose.total_size
        assert array.total_bytes == array_transpose.total_bytes
        _check_arrays_equal(array, array_transpose.transpose())

    _check_transpose(array.transpose())
    _check_transpose(array.T)
    _check_transpose(xchainer.transpose(array))


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
    a_xc = xchainer.Array(a_np)

    def check(b_xc):
        assert b_xc is not a_xc
        assert b_np.shape == b_xc.shape
        assert b_xc.is_contiguous
        assert a_xc._debug_data_memory_address == b_xc._debug_data_memory_address, 'Reshape must be done without copy'
        assert b_xc.strides == b_np.strides, 'Strides after reshape must match NumPy behavior'
        _check_arrays_equal(xchainer.Array(b_np), b_xc)

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


def test_reshape_backward():
    x = xchainer.Array(numpy.arange(6, dtype=numpy.float32)).require_grad()
    gy = xchainer.ones((2, 3), x.dtype)
    eps = xchainer.full_like(x, 1e-3)
    xchainer.check_backward(lambda a: (a[0].reshape(gy.shape),), [x], [gy], [eps])


def test_reshape_double_backward():
    x = xchainer.Array(numpy.arange(6, dtype=numpy.float32)).require_grad()
    gy = xchainer.ones((2, 3), x.dtype).require_grad()
    ggx = xchainer.ones_like(x)
    eps_x = xchainer.full_like(x, 1e-3)
    eps_gy = xchainer.full_like(gy, 1e-3)

    def forward(a):
        b = a[0].reshape(gy.shape)
        return b * b,  # to make it nonlinear

    xchainer.check_double_backward(forward, [x], [gy], [ggx], [eps_x, eps_gy], atol=1e-4)


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
        a_xc = xchainer.Array(a_np)

        with pytest.raises(xchainer.DimensionError):
            a_xc.reshape(b_shape)

    check(shape1, shape2)
    check(shape2, shape1)


@pytest.mark.parametrize('src_shape,dst_shape', [
    ((), ()),
    ((1,), (2,)),
    ((1, 1), (2, 2)),
    ((1, 1), (1, 2)),
])
def test_broadcast_to(src_shape, dst_shape):
    size = functools.reduce(operator.mul, src_shape, 1)
    src_np = numpy.arange(size, dtype=numpy.float32).reshape(src_shape)
    src = xchainer.Array(src_np)

    dst = xchainer.broadcast_to(src, dst_shape)
    dst_np = numpy.broadcast_to(src_np, dst_shape)
    _check_array_equals_ndarray(dst, dst_np)


def test_broadcast_to_auto_prefix():
    src_np = numpy.arange(2, dtype=numpy.float32)
    src = xchainer.Array(src_np)

    dst_np = numpy.broadcast_to(src_np, (3, 2))
    dst = xchainer.broadcast_to(src, (3, 2))

    _check_array_equals_ndarray(dst, dst_np)


@pytest.mark.parametrize(('src_shape,dst_shape'), [
    ((3,), (2,)),
    ((3,), (3, 2)),
    ((1, 3), (3, 2)),
])
def test_invalid_broadcast_to(src_shape, dst_shape):
    src = xchainer.ones(src_shape, xchainer.float32)
    with pytest.raises(xchainer.DimensionError):
        xchainer.broadcast_to(src, dst_shape)


def test_broadcast_to_backward():
    x = xchainer.Array(numpy.arange(9, dtype=numpy.float32).reshape(1, 3, 1, 3)).require_grad()
    gy = xchainer.ones((2, 3, 4, 3), x.dtype)
    eps = xchainer.full_like(x, 1e-2)
    xchainer.check_backward(lambda a: (xchainer.broadcast_to(a[0], (2, 3, 4, 3)), ), [x], [gy], [eps], atol=1e-4)


def test_broadcast_to_double_backward():
    x = xchainer.Array(numpy.arange(9, dtype=numpy.float32).reshape(1, 3, 1, 3)).require_grad()
    gy = xchainer.ones((2, 3, 4, 3), x.dtype).require_grad()
    ggx = xchainer.ones_like(x)
    eps_x = xchainer.full_like(x, 1e-2)
    eps_gy = xchainer.full_like(gy, 1e-2)

    def forward(a):
        b = xchainer.broadcast_to(a[0], (2, 3, 4, 3))
        return b * b,  # to make it nonlinear

    xchainer.check_double_backward(forward, [x], [gy], [ggx], [eps_x, eps_gy], atol=1e-3)


def test_copy(array_init_inputs):
    shape, dtype = array_init_inputs
    data_list = _create_dummy_data(shape, dtype)
    array = xchainer.Array(shape, dtype, data_list)

    _check_arrays_equal_copy(array, array.copy())
    _check_arrays_equal_copy(array, xchainer.copy(array))


def test_as_constant_copy(array_init_inputs):
    shape, dtype = array_init_inputs
    data_list = _create_dummy_data(shape, dtype)

    # Stop gradients on all graphs
    a = xchainer.Array(shape, dtype, data_list)
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
    a = xchainer.Array(shape, dtype, data_list)
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


def test_as_constant_view(array_init_inputs):
    shape, dtype = array_init_inputs
    data_list = _create_dummy_data(shape, dtype)

    # Stop gradients on all graphs
    a = xchainer.Array(shape, dtype, data_list)
    a.require_grad('graph_1')
    a.require_grad('graph_2')
    assert a.is_grad_required('graph_1')
    assert a.is_grad_required('graph_2')
    b = a.as_constant(copy=False)

    _check_array(b, dtype, shape, _size(shape), data_list)
    assert not b.is_grad_required('graph_1')
    assert not b.is_grad_required('graph_2')

    assert a.is_grad_required('graph_1')
    assert a.is_grad_required('graph_2')

    # Stop gradients on some graphs
    a = xchainer.Array(shape, dtype, data_list)
    a.require_grad('graph_1')
    a.require_grad('graph_2')
    a.require_grad('graph_3')
    assert a.is_grad_required('graph_1')
    assert a.is_grad_required('graph_2')
    assert a.is_grad_required('graph_3')
    b = a.as_constant(['graph_1', 'graph_2'], copy=False)

    _check_array(b, dtype, shape, _size(shape), data_list)
    assert not b.is_grad_required('graph_1')
    assert not b.is_grad_required('graph_2')
    assert b.is_grad_required('graph_3')

    assert a.is_grad_required('graph_1')
    assert a.is_grad_required('graph_2')
    assert a.is_grad_required('graph_3')


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_add_iadd(device, array_init_inputs):
    shape, dtype = array_init_inputs

    lhs_data_list = _create_dummy_data(shape, dtype, pattern=1)
    rhs_data_list = _create_dummy_data(shape, dtype, pattern=2)

    lhs = xchainer.Array(shape, dtype, lhs_data_list)
    rhs = xchainer.Array(shape, dtype, rhs_data_list)

    expected_data_list = [x + y for x, y in zip(lhs_data_list, rhs_data_list)]
    if dtype == xchainer.Dtype.bool:
        expected_data_list = [x > 0 for x in expected_data_list]  # [0, 2] => [False, True]

    def _check_add(lhs, rhs, out):
        assert out._debug_flat_data == expected_data_list
        assert lhs._debug_flat_data == lhs_data_list
        assert rhs._debug_flat_data == rhs_data_list

    _check_add(lhs, rhs, lhs + rhs)
    _check_add(lhs, rhs, xchainer.add(lhs, rhs))

    lhs_prev = lhs
    lhs += rhs
    assert lhs is lhs_prev, 'inplace operation must not alter lhs reference'
    assert lhs._debug_flat_data == expected_data_list
    assert rhs._debug_flat_data == rhs_data_list


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_mul_imul(device, array_init_inputs):
    shape, dtype = array_init_inputs

    lhs_data_list = _create_dummy_data(shape, dtype, pattern=1)
    rhs_data_list = _create_dummy_data(shape, dtype, pattern=2)

    lhs = xchainer.Array(shape, dtype, lhs_data_list)
    rhs = xchainer.Array(shape, dtype, rhs_data_list)

    expected_data_list = [x * y for x, y in zip(lhs_data_list, rhs_data_list)]
    if dtype == xchainer.Dtype.bool:
        expected_data_list = [x > 0 for x in expected_data_list]  # [0, 1] => [False, True]

    def _check_mul(lhs, rhs, out):
        assert out._debug_flat_data == expected_data_list
        assert lhs._debug_flat_data == lhs_data_list
        assert rhs._debug_flat_data == rhs_data_list

    _check_mul(lhs, rhs, lhs * rhs)
    _check_mul(lhs, rhs, xchainer.multiply(lhs, rhs))

    lhs_prev = lhs
    lhs *= rhs
    assert lhs is lhs_prev, 'inplace operation must not alter lhs reference'
    assert lhs._debug_flat_data == expected_data_list
    assert rhs._debug_flat_data == rhs_data_list


@pytest.mark.parametrize('scalar', [0, -1, 1, 2])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_mul_scalar(scalar, device, array_init_inputs):
    shape, dtype = array_init_inputs

    data_list = _create_dummy_data(shape, dtype)

    # Implicit casting in NumPy's multiply depends on the 'casting' argument,
    # which is not yet supported (xChainer always casts).
    # Therefore, we explicitly cast the scalar to the dtype of the ndarray
    # before the multiplication for NumPy.
    scalar_np = numpy.dtype(dtype.name).type(scalar)
    expected = numpy.array(data_list, dtype=dtype.name).reshape(shape)
    expected *= scalar_np

    x = xchainer.Array(shape, dtype, data_list)
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
def test_mul_backward(device):
    x1 = xchainer.Array(numpy.arange(6, dtype=numpy.float32).reshape(2, 3)).require_grad()
    x2 = xchainer.Array(numpy.arange(3, dtype=numpy.float32).reshape(3)).require_grad()
    gout = xchainer.ones_like(x1)
    eps_x1 = xchainer.full_like(x1, 1e-2)
    eps_x2 = xchainer.full_like(x2, 1e-2)
    xchainer.check_backward(lambda xs: (xs[0] * xs[1],), [x1, x2], [gout], [eps_x1, eps_x2])
    xchainer.check_backward(lambda xs: (xchainer.multiply(xs[0], xs[1]),), [x1, x2], [gout], [eps_x1, eps_x2])


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_mul_scalar_backward(device):
    x1 = xchainer.Array(numpy.arange(6, dtype=numpy.float32).reshape(2, 3)).require_grad()
    gout = xchainer.ones_like(x1)
    eps_x1 = xchainer.full_like(x1, 1e-2)
    xchainer.check_backward(lambda xs: (xs[0] * 3.2,), [x1], [gout], [eps_x1])
    xchainer.check_backward(lambda xs: (xchainer.multiply(xs[0], 3.2),), [x1], [gout], [eps_x1])


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_mul_double_backward(device):
    x1 = xchainer.Array(numpy.arange(6, dtype=numpy.float32).reshape(2, 3)).require_grad()
    x2 = xchainer.Array(numpy.arange(3, dtype=numpy.float32).reshape(3)).require_grad()
    gout = xchainer.ones_like(x1).require_grad()
    ggx1 = xchainer.ones_like(x1)
    ggx2 = xchainer.ones_like(x2)
    eps_x1 = xchainer.full_like(x1, 1e-2)
    eps_x2 = xchainer.full_like(x2, 1e-2)
    eps_gout = xchainer.full_like(gout, 1e-2)

    def check(func):
        xchainer.check_double_backward(func, [x1, x2], [gout], [ggx1, ggx2], [eps_x1, eps_x2, eps_gout], atol=1e-4)

    check(lambda xs: (xs[0] * xs[1],))
    check(lambda xs: (xchainer.multiply(xs[0], xs[1]),))


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_mul_scalar_double_backward(device):
    x1 = xchainer.Array(numpy.arange(6, dtype=numpy.float32).reshape(2, 3)).require_grad()
    gout = xchainer.ones_like(x1).require_grad()
    ggx1 = xchainer.ones_like(x1)
    eps_x1 = xchainer.full_like(x1, 1e-2)
    eps_gout = xchainer.full_like(gout, 1e-2)

    def check(func):
        xchainer.check_double_backward(func, [x1], [gout], [ggx1], [eps_x1, eps_gout], atol=1e-4)

    def forward_operator(xs):
        out = xs[0] * 1.2
        return out * out,  # to make it nonlinear

    def forward_function(xs):
        out = xchainer.multiply(-0.3, xs[0])
        return out * out,  # to make it nonlinear

    check(forward_operator)
    check(forward_function)


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
    assert "array([], shape=(0,), dtype=bool, device='native:0')" == str(array)

    array = xchainer.Array((1,), xchainer.Dtype.bool, [False])
    assert "array([False], shape=(1,), dtype=bool, device='native:0')" == str(array)

    array = xchainer.Array((2, 3), xchainer.Dtype.int8, [0, 1, 2, 3, 4, 5])
    assert ("array([[0, 1, 2],\n"
            "       [3, 4, 5]], shape=(2, 3), dtype=int8, device='native:0')") == str(array)

    array = xchainer.Array((2, 3), xchainer.Dtype.float32, [0, 1, 2, 3.25, 4, 5])
    assert ("array([[0.  , 1.  , 2.  ],\n"
            "       [3.25, 4.  , 5.  ]], shape=(2, 3), dtype=float32, device='native:0')") == str(array)


def test_array_require_grad():
    array = xchainer.Array((3, 1), xchainer.Dtype.int8, [1, 1, 1])

    assert not array.is_grad_required()
    array.require_grad()
    assert array.is_grad_required()

    with pytest.raises(xchainer.XchainerError):
        array.require_grad()


def test_array_require_grad_with_graph_id():
    array = xchainer.Array((3, 1), xchainer.Dtype.int8, [1, 1, 1])

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
    array = xchainer.Array((3, 1), xchainer.Dtype.int8, [1, 1, 1])
    grad = xchainer.Array((3, 1), xchainer.Dtype.float32, [0.5, 0.5, 0.5])

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
    array = xchainer.Array((3, 1), xchainer.Dtype.int8, [1, 1, 1])
    grad = xchainer.Array((3, 1), xchainer.Dtype.float32, [0.5, 0.5, 0.5])

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
    array = xchainer.Array(shape, dtype, [2, 5, 1])
    grad = xchainer.Array(shape, dtype, [5, 7, 8])

    # Set grad
    array.require_grad().set_grad(grad)

    # Retrieve grad twice and assert they share the same underlying data
    grad1 = array.get_grad()
    grad2 = array.get_grad()

    grad1 *= xchainer.Array(shape, dtype, [2, 2, 2])
    assert grad2._debug_flat_data == [10, 14, 16], 'grad getter must not incur a copy'


def test_array_cleargrad():
    shape = (3, 1)
    dtype = xchainer.int8
    array = xchainer.Array(shape, dtype, [2, 5, 1])
    grad = xchainer.Array(shape, dtype, [5, 7, 8])

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
    array = xchainer.Array(shape, xchainer.int8, [1, 1, 1])
    grad = xchainer.Array(shape, xchainer.float32, [0.5, 0.5, 0.5])
    array.require_grad().set_grad(grad)

    assert array.get_grad() is grad, 'grad must preserve physical identity'
    assert array.get_grad() is grad, 'grad must preserve physical identity in repeated retrieval'

    # array.grad and grad share the same data
    grad += xchainer.Array(shape, xchainer.float32, [2, 2, 2])
    assert array.get_grad()._debug_flat_data == [2.5, 2.5, 2.5], 'A modification to grad must affect array.grad'

    array_grad = array.get_grad()
    array_grad += xchainer.Array(shape, xchainer.float32, [1, 1, 1])
    assert grad._debug_flat_data == [3.5, 3.5, 3.5], 'A modification to array.grad must affect grad'


def test_array_require_grad_multiple_graphs_forward():
    x1 = xchainer.Array((3, 1), xchainer.Dtype.int8, [1, 1, 1])
    x2 = xchainer.Array((3, 1), xchainer.Dtype.int8, [1, 1, 1])

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
    x1 = xchainer.Array((3, 1), xchainer.int8, [1, 1, 1]).require_grad(graph_id='graph_1')
    x2 = xchainer.Array((3, 1), xchainer.int8, [1, 1, 1]).require_grad(graph_id='graph_1')
    y = x1 * x2

    y.backward(graph_id='graph_1', enable_double_backprop=True)
    gx1 = x1.get_grad(graph_id='graph_1')
    x1.set_grad(None, graph_id='graph_1')

    gx1.backward(graph_id='graph_1')
    assert gx1.get_grad(graph_id='graph_1') is not None


@pytest.mark.parametrize("input_shape,indices,output_shape,output_data", [
    # empty indexing
    ((), (), (), [0]),
    ((3,), (), (3,), [0, 1, 2]),
    ((2, 2, 2), (), (2, 2, 2), [0, 1, 2, 3, 4, 5, 6, 7]),
    # integer indexing - non-tuple indexing
    ((3,), 0, (), [0]),
    ((3,), 1, (), [1]),
    ((3,), 2, (), [2]),
    ((3,), -1, (), [2]),
    ((2, 3), 0, (3,), [0, 1, 2]),
    ((2, 3), 1, (3,), [3, 4, 5]),
    # integer indexining - tuple indexing
    ((3,), (0,), (), [0]),
    ((3,), (1,), (), [1]),
    ((3,), (2,), (), [2]),
    ((3,), (-1,), (), [2]),
    ((2, 3), (0,), (3,), [0, 1, 2]),
    ((2, 3), (1,), (3,), [3, 4, 5]),
    ((2, 3), (0, 0), (), [0]),
    ((2, 3), (1, 1), (), [4]),
    ((2, 3, 4), (0, -2, 3), (), [7]),
    ((2, 3, 4), (1, 0), (4,), [12, 13, 14, 15]),
    # slice indexing - non-tuple indexing
    ((3,), slice(None), (3,), [0, 1, 2]),
    ((3,), slice(2), (2,), [0, 1]),
    ((3,), slice(0, 3), (3,), [0, 1, 2]),
    ((3,), slice(0, 2), (2,), [0, 1]),
    ((3,), slice(1, 3), (2,), [1, 2]),
    ((3,), slice(0, 0), (0,), []),
    ((3,), slice(0, 1), (1,), [0]),
    ((3,), slice(2, 0, -1), (2,), [2, 1]),
    ((3,), slice(-2, -1), (1,), [1]),
    ((3,), slice(2, None, -1), (3,), [2, 1, 0]),
    ((3,), slice(None, 0, 1), (0,), []),
    ((3,), slice(None, -1, -1), (0,), []),
    ((3,), slice(None, -2, -1), (1,), [2]),
    ((6,), slice(0, 6, 2), (3,), [0, 2, 4]),
    ((6,), slice(1, 6, 2), (3,), [1, 3, 5]),
    ((6,), slice(5, None, -2), (3,), [5, 3, 1]),
    # slice indexing - tuple indexing
    ((3,), (slice(None),), (3,), [0, 1, 2]),
    ((3,), (slice(2),), (2,), [0, 1]),
    ((3,), (slice(0, 3),), (3,), [0, 1, 2]),
    ((3,), (slice(0, 2),), (2,), [0, 1]),
    ((3,), (slice(1, 3),), (2,), [1, 2]),
    ((3,), (slice(0, 0),), (0,), []),
    ((3,), (slice(0, 1),), (1,), [0]),
    ((3,), (slice(2, 0, -1),), (2,), [2, 1]),
    ((3,), (slice(-2, -1),), (1,), [1]),
    ((3,), (slice(2, None, -1),), (3,), [2, 1, 0]),
    ((3,), (slice(None, 0, 1),), (0,), []),
    ((3,), (slice(None, -1, -1),), (0,), []),
    ((3,), (slice(None, -2, -1),), (1,), [2]),
    ((6,), (slice(0, 6, 2),), (3,), [0, 2, 4]),
    ((6,), (slice(1, 6, 2),), (3,), [1, 3, 5]),
    ((6,), (slice(5, None, -2),), (3,), [5, 3, 1]),
    ((2, 3), (slice(None), slice(None)), (2, 3), [0, 1, 2, 3, 4, 5]),
    ((2, 3), (slice(1), slice(2)), (1, 2), [0, 1]),
    ((2, 3), (slice(0, 2), slice(0, 3)), (2, 3), [0, 1, 2, 3, 4, 5]),
    ((2, 3), (slice(0, 2), slice(0, -1)), (2, 2), [0, 1, 3, 4]),
    ((2, 3), (slice(0, None, -1), slice(2, 3)), (1, 1), [2]),
    ((2, 3), (slice(0, None, None), slice(-2, 0, -1)), (2, 1), [1, 4]),
    ((2, 3), (slice(1, 2), slice(0, 2)), (1, 2), [3, 4]),
    ((2, 3), (slice(-2, None, -1), slice(0, 3)), (1, 3), [0, 1, 2]),
    ((2, 3), (slice(-2, None, -1), slice(-3, None, -1)), (1, 1), [0]),
    ((2, 3), (slice(-2, None, -1), slice(None, None, -2)), (1, 2), [2, 0]),
    ((2, 3), (slice(1, 2), slice(None, None, 1)), (1, 3), [3, 4, 5]),
    ((2, 3), (slice(1, 2), slice(None, None, 2)), (1, 2), [3, 5]),
    ((2, 3, 4), (slice(1), slice(-2, 3), slice(1, None, -1)), (1, 2, 2), [5, 4, 9, 8]),
    # newaxis indexing - non-tuple indexing
    ((), xchainer.newaxis, (1,), [0]),
    ((3,), xchainer.newaxis, (1, 3), [0, 1, 2]),
    # newaxis indexing - tuple indexing
    ((), (xchainer.newaxis,), (1,), [0]),
    ((3,), (xchainer.newaxis,), (1, 3), [0, 1, 2]),
    ((2, 3), (xchainer.newaxis, xchainer.newaxis), (1, 1, 2, 3), [0, 1, 2, 3, 4, 5]),
    # mixed indexing - tuple indexing
    ((2, 3), (0, slice(1, 3)), (2,), [1, 2]),
    ((4, 3), (slice(1, 3), 1), (2,), [4, 7]),
    ((2, 3, 4), (1, slice(2,), slice(1, 3)), (2, 2), [13, 14, 17, 18]),
    ((2, 3), (1, xchainer.newaxis, slice(1, 3)), (1, 2), [4, 5]),
    ((2, 3, 4), (slice(0, 1), slice(1, 2), slice(1, 3), xchainer.newaxis), (1, 1, 2, 1), [5, 6]),
    ((2, 3, 4), (slice(0, 1), slice(1, 2), xchainer.newaxis, slice(1, 3)), (1, 1, 1, 2), [5, 6]),
    ((2, 3, 4), (slice(0, 1), xchainer.newaxis, slice(1, 2), slice(1, 3)), (1, 1, 1, 2), [5, 6]),
    ((2, 3, 4), (xchainer.newaxis, slice(0, 1), slice(1, 2), slice(1, 3)), (1, 1, 1, 2), [5, 6]),
    ((2, 3, 4), (1, slice(2,), xchainer.newaxis, slice(1, 3), xchainer.newaxis), (2, 1, 2, 1), [13, 14, 17, 18]),
])
def test_getitem(input_shape, indices, output_shape, output_data):
    total_size = functools.reduce(operator.mul, input_shape, 1)
    input_data = list(range(0, total_size))
    x = xchainer.Array(input_shape, xchainer.int32, input_data)
    y = x[indices]
    e = xchainer.Array(output_shape, xchainer.int32, output_data)
    _check_arrays_equal(y, e)

    n = numpy.array(input_data, numpy.int32).reshape(input_shape)
    _check_array_equals_ndarray(y, n[indices])


@pytest.mark.parametrize("input_shape,axis,keepdims,output_shape,output_data", [
    ((), None, False, (), [0]),
    ((), None, True, (), [0]),
    ((), (), False, (), [0]),
    ((), (), True, (), [0]),
    ((2,), None, False, (), [1]),
    ((2,), None, True, (1,), [1]),
    ((2,), (), False, (2,), [0, 1]),
    ((2,), (), True, (2,), [0, 1]),
    ((2,), 0, False, (), [1]),
    ((2,), 0, True, (1,), [1]),
    ((2,), (0,), False, (), [1]),
    ((2,), (0,), True, (1,), [1]),
    ((2,), (-1,), False, (), [1]),
    ((2,), (-1,), True, (1,), [1]),
    ((2, 3), None, False, (), [15]),
    ((2, 3), None, True, (1, 1), [15]),
    ((2, 3), (), False, (2, 3), [0, 1, 2, 3, 4, 5]),
    ((2, 3), (), True, (2, 3), [0, 1, 2, 3, 4, 5]),
    ((2, 3), 0, False, (3,), [3, 5, 7]),
    ((2, 3), 0, True, (1, 3), [3, 5, 7]),
    ((2, 3), (0,), False, (3,), [3, 5, 7]),
    ((2, 3), (0,), True, (1, 3), [3, 5, 7]),
    ((2, 3), (1,), False, (2,), [3, 12]),
    ((2, 3), (1,), True, (2, 1), [3, 12]),
    ((2, 3), (-1,), False, (2,), [3, 12]),
    ((2, 3), (-1,), True, (2, 1), [3, 12]),
    ((2, 3), (-2,), False, (3,), [3, 5, 7]),
    ((2, 3), (-2,), True, (1, 3), [3, 5, 7]),
    ((2, 3), (0, 1), False, (), [15]),
    ((2, 3), (0, 1), True, (1, 1), [15]),
    ((2, 3), (-2, -1), False, (), [15]),
    ((2, 3), (-2, -1), True, (1, 1), [15]),

    # Sum over axes that are in the middle or apart
    ((2, 3, 4), (1,), False, (2, 4), [12, 15, 18, 21, 48, 51, 54, 57]),
    ((2, 3, 4), (1,), True, (2, 1, 4), [12, 15, 18, 21, 48, 51, 54, 57]),
    ((2, 3, 4), (0, 2), False, (3,), [60, 92, 124]),
    ((2, 3, 4), (0, 2), True, (1, 3, 1), [60, 92, 124]),

    # Sum over axes that are apart and/or unsorted
    ((2, 3), (1, 0), False, (), [15]),
    ((2, 3), (1, 0), True, (1, 1), [15]),
    ((2, 3, 4), (2, 0), False, (3,), [60, 92, 124]),
    ((2, 3, 4), (2, 0), True, (1, 3, 1), [60, 92, 124]),
    ((2, 3, 4), (2, 0, 1), False, (), [276]),
    ((2, 3, 4), (2, 0, 1), True, (1, 1, 1), [276]),
    ((2, 3, 4), (-2, 2, 0), False, (), [276]),
    ((2, 3, 4), (-2, 2, 0), True, (1, 1, 1), [276]),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_sum(device, input_shape, axis, keepdims, output_shape, output_data):
    total_size = functools.reduce(operator.mul, input_shape, 1)
    input_data = list(range(0, total_size))
    x = xchainer.Array(input_shape, xchainer.int32, input_data)
    e = xchainer.Array(output_shape, xchainer.int32, output_data)
    n = numpy.array(input_data, numpy.int32).reshape(input_shape).sum(axis=axis, keepdims=keepdims).astype(numpy.int32)

    y = x.sum(axis=axis, keepdims=keepdims)
    _check_arrays_equal(y, e)
    _check_array_equals_ndarray(y, n)

    y = xchainer.sum(x, axis, keepdims)
    _check_arrays_equal(y, e)
    _check_array_equals_ndarray(y, n)


@pytest.mark.parametrize("input_shape,axis,keepdims,error", [
    ((), 0, False, xchainer.DimensionError),
    ((), 0, True, xchainer.DimensionError),
    ((), 1, False, xchainer.DimensionError),
    ((), 1, True, xchainer.DimensionError),
    ((), (1,), False, xchainer.DimensionError),
    ((), (1,), True, xchainer.DimensionError),
    ((2,), 2, False, xchainer.DimensionError),
    ((2,), 2, True, xchainer.DimensionError),
    ((2,), (2,), False, xchainer.DimensionError),
    ((2,), (2,), True, xchainer.DimensionError),
    ((2,), (-2,), False, xchainer.DimensionError),
    ((2,), (-2,), True, xchainer.DimensionError),
    ((2, 3,), (-3,), False, xchainer.DimensionError),
    ((2, 3,), (-3,), True, xchainer.DimensionError),
    ((2, 3,), (-3, -4), False, xchainer.DimensionError),
    ((2, 3,), (-3, -4), True, xchainer.DimensionError),
    ((2, 3,), (0, 0), False, xchainer.XchainerError),
    ((2, 3,), (0, 0), True, xchainer.XchainerError),
    ((2, 3,), (-1, -1), False, xchainer.XchainerError),
    ((2, 3,), (-1, -1), True, xchainer.XchainerError),
    ((2, 3,), (0, 1, 1), False, xchainer.XchainerError),
    ((2, 3,), (0, 1, 1), True, xchainer.XchainerError),
    ((2, 3,), (0, -2), False, xchainer.XchainerError),
    ((2, 3,), (0, -2), True, xchainer.XchainerError),
])
def test_invalid_sum(input_shape, axis, keepdims, error):
    total_size = functools.reduce(operator.mul, input_shape, 1)
    input_data = list(range(0, total_size))
    x = xchainer.Array(input_shape, xchainer.int32, input_data)

    with pytest.raises(error):
        x.sum(axis=axis, keepdims=keepdims)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_sum_backward(device):
    x = xchainer.Array(numpy.arange(6, dtype=numpy.float32).reshape(2, 3)).require_grad()
    gy = xchainer.ones((2,), x.dtype)
    eps = xchainer.full_like(x, 1e-2)
    xchainer.check_backward(lambda a: (a[0].sum(axis=1),), [x], [gy], [eps])


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_sum_double_backward(device):
    x = xchainer.Array(numpy.arange(6, dtype=numpy.float32).reshape(2, 3)).require_grad()
    gy = xchainer.ones((2,), x.dtype).require_grad()
    ggx = xchainer.ones_like(x)
    eps_x = xchainer.full_like(x, 1e-2)
    eps_gy = xchainer.full_like(gy, 1e-2)

    def forward(a):
        b = a[0].sum(axis=1)
        return b * b,  # to make it nonlinear

    xchainer.check_double_backward(forward, [x], [gy], [ggx], [eps_x, eps_gy], atol=1e-4)
