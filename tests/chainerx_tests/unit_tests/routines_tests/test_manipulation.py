import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils


# Value for parameterization to represent an unspecified (default) argument.
class _UnspecifiedType(object):
    def __repr__(self):
        return '<Unspecified>'


_unspecified = _UnspecifiedType()


@chainerx.testing.numpy_chainerx_array_equal()
def test_transpose(is_module, xp, shape, dtype):
    array = array_utils.create_dummy_ndarray(xp, shape, dtype)
    if is_module:
        return xp.transpose(array)
    else:
        return array.transpose()


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('shape,axes', [
    ((1,), 0),
    ((1,), (0,)),
    ((2,), (0,)),
    ((2, 3), (1, 0)),
    ((2, 3), (-2, -1)),
    ((2, 3, 1), (2, 0, 1)),
    ((2, 3, 1), (2, -3, 1)),
])
def test_transpose_axes(is_module, xp, shape, axes, dtype):
    array = array_utils.create_dummy_ndarray(xp, shape, dtype)
    if is_module:
        return xp.transpose(array, axes)
    else:
        return array.transpose(axes)


@pytest.mark.parametrize('shape,axes', [
    ((), (0,)),
    ((1,), (1,)),
    ((2, 3), (1,)),
    ((2, 3), (1, 0, 2)),
])
def test_transpose_invalid_axes(shape, axes):
    a = array_utils.create_dummy_ndarray(chainerx, shape, 'float32')
    with pytest.raises(chainerx.DimensionError):
        chainerx.transpose(a, axes)
    with pytest.raises(chainerx.DimensionError):
        a.transpose(axes)


@chainerx.testing.numpy_chainerx_array_equal()
def test_T(xp, shape, dtype):
    array = array_utils.create_dummy_ndarray(xp, shape, dtype)
    return array.T


_reshape_shape = [
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
    ((2, 3, 4), (3, -1, 2)),
    ((2, 3, 4), (3, -3, 2)),  # -3 is treated as a -1 and is valid.
]


# TODO(niboshi): Test with non-contiguous input array that requires copy to
# reshape
# TODO(niboshi): Test with non-contiguous input array that does not require
# copy to reshape
@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('a_shape,b_shape', _reshape_shape)
@pytest.mark.parametrize('shape_type', [tuple, list])
@pytest.mark.parametrize('padding', [False, True])
def test_reshape(is_module, xp, a_shape, b_shape, shape_type, padding):
    a = array_utils.create_dummy_ndarray(xp, a_shape, 'int64', padding=padding)
    if is_module:
        b = xp.reshape(a, shape_type(b_shape))
    else:
        b = a.reshape(shape_type(b_shape))

    if xp is chainerx:
        copied = a._debug_data_memory_address != b._debug_data_memory_address
    else:
        copied = a.ctypes.data != b.ctypes.data

    if copied:
        if xp is chainerx:
            assert b.is_contiguous
        else:
            assert b.flags.c_contiguous

    return copied, b


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(TypeError, chainerx.ChainerxError))
@pytest.mark.parametrize('a_shape,b_shape', _reshape_shape)
def test_reshape_args(is_module, xp, a_shape, b_shape):
    # TODO(niboshi): Remove padding=False
    a = array_utils.create_dummy_ndarray(xp, a_shape, 'int64', padding=False)
    if is_module:
        if len(b_shape) > 1:
            # Skipping tests where the 'order' argument is unintentionally
            # given a shape value, since numpy won't raise any errors in this
            # case which you might expect at first.
            return xp.array([])
        # TypeError/chainerx.ChainerxError in case b_shape is empty.
        b = xp.reshape(a, *b_shape)
    else:
        # TypeError/chainerx.ChainerxError in case b_shape is empty.
        b = a.reshape(*b_shape)

    if xp is chainerx:
        assert b.is_contiguous
        assert a._debug_data_memory_address == b._debug_data_memory_address, (
            'Reshape must be done without copy')
        assert numpy.arange(a.size).reshape(b_shape).strides == b.strides, (
            'Strides after reshape must match NumPy behavior')

    return b


@pytest.mark.parametrize('shape1,shape2', [
    ((), (0,)),
    ((), (2,)),
    ((), (1, 2,)),
    ((0,), (1,)),
    ((0,), (1, 1, 1)),
    ((2, 3), (2, 3, 2)),
    ((2, 3, 4), (2, 3, 5)),
])
def test_reshape_invalid(shape1, shape2):
    def check(a_shape, b_shape):
        a = array_utils.create_dummy_ndarray(chainerx, a_shape, 'float32')
        with pytest.raises(chainerx.DimensionError):
            a.reshape(b_shape)

    check(shape1, shape2)
    check(shape2, shape1)


@pytest.mark.parametrize('shape1,shape2', [
    ((2, 3, 4), (5, -1, 3)),  # Not divisible.
    ((2, 3, 4), (-1, -1, 3)),  # More than one dimension cannot be inferred.
    ((2, 3, 4), (-2, 4, -1)),
])
def test_reshape_invalid_cannot_infer(shape1, shape2):
    a = array_utils.create_dummy_ndarray(chainerx, shape1, 'float32')
    with pytest.raises(chainerx.DimensionError):
        a.reshape(shape2)


@chainerx.testing.numpy_chainerx_array_equal()
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
    a = array_utils.create_dummy_ndarray(xp, shape, 'float32')
    if is_module:
        return xp.squeeze(a, axis)
    else:
        return a.squeeze(axis)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(chainerx.DimensionError, ValueError))
@pytest.mark.parametrize('shape,axis', [
    ((2, 1, 3), 0),
    ((2, 1, 3), -1),
    ((2, 1, 3), (1, 2)),
    ((2, 1, 3), (1, -1)),
    ((2, 1, 3), (1, 1)),
])
def test_squeeze_invalid(is_module, xp, shape, axis):
    a = xp.ones(shape, 'float32')
    if is_module:
        return xp.squeeze(a, axis)
    else:
        return a.squeeze(axis)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('src_shape,dst_shape', [
    ((), ()),
    ((1,), (2,)),
    ((1, 1), (2, 2)),
    ((1, 1), (1, 2)),
])
def test_broadcast_to(xp, src_shape, dst_shape):
    a = array_utils.create_dummy_ndarray(xp, src_shape, 'float32')
    return xp.broadcast_to(a, dst_shape)


@chainerx.testing.numpy_chainerx_array_equal()
def test_broadcast_to_auto_prefix(xp):
    a = xp.arange(2, dtype='float32')
    return xp.broadcast_to(a, (3, 2))


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(chainerx.DimensionError, ValueError))
@pytest.mark.parametrize(('src_shape,dst_shape'), [
    ((3,), (2,)),
    ((3,), (3, 2)),
    ((1, 3), (3, 2)),
])
def test_broadcast_to_invalid(xp, src_shape, dst_shape):
    a = xp.ones(src_shape, 'float32')
    return xp.broadcast_to(a, dst_shape)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(chainerx.DimensionError, ValueError))
@pytest.mark.parametrize('shapes,axis', [
    ([], 0),
    ([(0,)], 0),
    ([(1,)], 0),
    ([(0,), (0,)], 0),
    ([(0,), (1,)], 0),
    ([(1,), (1,)], 0),
    ([(0, 0,), (0, 0,)], 0),
    ([(0, 0,), (0, 0,)], 1),
    ([(1, 0,), (1, 0,)], 0),
    ([(1, 0,), (1, 0,)], 1),
    ([(1, 0,), (1, 0,)], 2),
    ([(3, 4, 5)], 0),
    ([(2, 3, 1), (2, 3, 1)], 1),
    ([(2, 3, 2), (2, 4, 2), (2, 3, 2)], 1),
    ([(2, 3, 2), (2, 4, 2), (3, 3, 2)], 1),
    ([(4, 10), (5, 10)], 0),
    ([(4, 10), (4, 8)], 0),
    ([(4, 4), (5,)], 0),
    ([(4, 4), (4,)], 0),
    ([(2, 3), (2, 3)], 10),
    ([(2, 3), (2, 3)], -1),
    ([(2, 3), (2, 3)], None),
    ([(2, 3), (4, 5)], None),
    ([(2, 3), (2, 3)], _unspecified),
    ([(2, 3), (4, 5)], _unspecified),
])
def test_concat(xp, shapes, axis):
    arrays = []
    for i, shape in enumerate(shapes):
        size = numpy.product(shape)
        a = numpy.arange(i * 100, i * 100 + size)
        a = a.reshape(shape).astype('float32')
        arrays.append(xp.array(a))
    if axis is _unspecified:
        args = (arrays,)
    else:
        args = (arrays, axis)
    return xp.concatenate(*args)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(chainerx.DimensionError, ValueError))
@pytest.mark.parametrize('shapes,axis', [
    ([], None),
    ([], 0),
    ([(0,)], -1),
    ([(0,)], 0),
    ([(0,)], 1),
    ([(0,)], 2),
    ([(1,)], -1),
    ([(1,)], 0),
    ([(1,)], 1),
    ([(1,)], 2),
    ([(0,), (0,)], 0),
    ([(0,), (0,)], 1),
    ([(0, 0,), (0, 0,)], 0),
    ([(0, 0,), (0, 0,)], 1),
    ([(1, 0,), (1, 0,)], 0),
    ([(1, 0,), (1, 0,)], 1),
    ([(1, 0,), (1, 0,)], 2),
    ([(3, 4, 5), (3, 4, 5), (3, 4, 5)], None),
    ([(3, 4, 5), (3, 4, 5), (3, 4, 5)], 0),
    ([(3, 4, 5), (3, 4, 5), (3, 4, 5)], 1),
    ([(3, 4, 5), (3, 4, 5), (3, 4, 5)], 2),
    ([(3, 4, 5), (3, 4, 5), (3, 4, 5)], 3),
    ([(3, 4, 5), (3, 4, 5), (3, 4, 5)], 4),
    ([(2, 3, 2), (2, 4, 2), (2, 3, 2)], 1),
])
def test_stack(xp, shapes, axis):
    arrays = []
    for i, shape in enumerate(shapes):
        size = numpy.product(shape)
        a = numpy.arange(i * 100, i * 100 + size)
        a = a.reshape(shape).astype('float32')
        arrays.append(xp.array(a))
    if axis is None:
        return xp.stack(arrays)
    else:
        return xp.stack(arrays, axis)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('shape,indices_or_sections,axis', [
    ((2,), 1, 0),
    ((2,), [], 0),
    ((2,), [1, 2], 0),
    ((2,), [-5, -3], 0),
    ((2, 4), 1, 0),
    ((2, 4), 2, 1),
    ((2, 4), 2, -1),
    ((2, 4, 6), [], 0),
    ((2, 4, 6), [2, 4], 2),
    ((2, 4, 6), [2, -3], 2),
    ((2, 4, 6), [2, 8], 2),
    ((2, 4, 6), [4, 2], 2),
    ((2, 4, 6), [1, 3], -2),
    ((6,), numpy.array([1, 2]), 0),  # indices with 1-d numpy array
    ((6,), numpy.array([2]), 0),  # indices with (1,)-shape numpy array
    ((6,), numpy.array(2), 0),  # sections numpy scalar
    ((6,), numpy.array(2.0), 0),  # sections with numpy scalar, float
    ((6,), 2.0, 0),  # float type sections, without fraction
    # indices with empty numpy indices
    ((6,), numpy.array([], numpy.int32), 0),
    ((6,), numpy.array([], numpy.float64), 0),
])
def test_split(xp, shape, indices_or_sections, axis):
    a = array_utils.create_dummy_ndarray(xp, shape, 'float32')
    return xp.split(a, indices_or_sections, axis)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(
        chainerx.DimensionError, IndexError, ValueError, TypeError,
        ZeroDivisionError))
@pytest.mark.parametrize('shape,indices_or_sections,axis', [
    ((), 1, 0),
    ((2,), 3, 0),
    ((2, 4), 0, 0),
    ((2, 4), -1, 1),
    ((2, 4), 1, 2),  # Axis out of range.
    ((2, 4), 3, 1),  # Uneven split.
    ((6,), [2.0], 0),  # float type indices
    ((6,), 2.1, 0),  # float type sections, with fraction
    # indices with (1,)-shape numpy array, float
    ((6,), numpy.array([2.0]), 0),
    # sections with numpy scalar, float with fraction
    ((6,), numpy.array(2.1), 0),
    ((2,), [1, 2.0], 0),  # indices with mixed type
    ((6,), '2', 0),  # Invalid type
    # indices with empty numpy indices
    ((6,), numpy.array([[], []], numpy.int32), 0),
    ((6,), numpy.array([[], []], numpy.float64), 0),
])
def test_split_invalid(xp, shape, indices_or_sections, axis):
    a = array_utils.create_dummy_ndarray(xp, shape, 'float32')
    return xp.split(a, indices_or_sections, axis)
