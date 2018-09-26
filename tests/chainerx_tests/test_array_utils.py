import numpy
import pytest

import chainerx

from chainerx_tests import array_utils


@pytest.mark.parametrize('expected,shape', [
    (1, ()),
    (1, (1,)),
    (0, (0,)),
    (0, (2, 0)),
    (2, (2,)),
    (6, (2, 3)),
])
def test_total_size(expected, shape):
    assert expected == array_utils.total_size(shape)


@pytest.mark.parametrize('xp', [numpy, chainerx])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('shape', [
    (),
    (0,),
    (1,),
    (2, 3),
    (1, 1, 1),
    (2, 0, 3),
])
@pytest.mark.parametrize('dtype', chainerx.testing.all_dtypes)
@pytest.mark.parametrize('pattern', [1, 2])
@pytest.mark.parametrize('padding', [True, False])
def test_dummy_ndarray(xp, device, shape, dtype, pattern, padding):
    a = array_utils.create_dummy_ndarray(
        xp, shape, dtype, device=device, pattern=pattern, padding=padding)

    assert isinstance(a, xp.ndarray)
    assert a.dtype == xp.dtype(dtype)
    assert a.shape == shape

    # Check values
    if xp is chainerx:
        a_np = chainerx.to_numpy(a)
    else:
        a_np = a
    if pattern == 1:
        if a.dtype.name == 'bool':
            expected_data = [i % 2 == 1 for i in range(a.size)]
        elif a.dtype.name in chainerx.testing.unsigned_dtypes:
            expected_data = list(range(a.size))
        else:
            expected_data = list(range(-1, a.size - 1))
    else:
        if a.dtype.name == 'bool':
            expected_data = [i % 3 == 0 for i in range(a.size)]
        elif a.dtype.name in chainerx.testing.unsigned_dtypes:
            expected_data = list(range(1, a.size + 1))
        else:
            expected_data = list(range(-2, a.size - 2))
    numpy.testing.assert_equal(a_np.ravel(), expected_data)

    # Check strides
    if xp is chainerx:
        assert a.device is device
    if not padding:
        if xp is chainerx:
            assert a.is_contiguous
        else:
            assert a.flags.c_contiguous


@pytest.mark.parametrize('device_spec', [None, 'native', 'native:0'])
def test_dummy_ndarray_device_spec(device_spec):
    a = array_utils.create_dummy_ndarray(
        chainerx, (2, 3), 'float32', device=device_spec)
    assert a.device is chainerx.get_device(device_spec)


@pytest.mark.parametrize('xp', [numpy, chainerx])
@pytest.mark.parametrize('shape,dtype,padding,expected_strides', [
    # padding=None means unspecified.
    ((), 'bool_', (), ()),
    ((), 'int32', (), ()),
    ((), 'int32', 1, ()),
    ((2,), 'bool_', (0,), (1,)),
    ((2,), 'bool_', (1,), (2,)),
    ((2,), 'bool_', (2,), (3,)),
    ((2,), 'bool_', 0, (1,)),
    ((2,), 'bool_', 1, (2,)),
    ((2,), 'int32', (0,), (4,)),
    ((2,), 'int32', (1,), (8,)),
    ((2,), 'int32', (2,), (12,)),
    ((2,), 'int32', 0, (4,)),
    ((2,), 'int32', 1, (8,)),
    ((0,), 'int32', (0,), (4,)),
    ((0,), 'int32', (1,), (8,)),
    ((0,), 'int32', (2,), (12,)),
    ((2, 3), 'bool_', (0, 0), (3, 1)),
    ((2, 3), 'bool_', (0, 1), (6, 2)),
    ((2, 3), 'bool_', (1, 1), (7, 2)),
    ((2, 3), 'bool_', (2, 3), (14, 4)),
    ((2, 3), 'bool_', 0, (3, 1)),
    ((2, 3), 'bool_', 1, (7, 2)),
    ((2, 3), 'int32', (0, 0,), (12, 4)),
    ((2, 3), 'int32', (0, 1,), (24, 8)),
    ((2, 3), 'int32', (1, 1,), (28, 8)),
    ((2, 3), 'int32', (2, 3,), (56, 16)),
    ((2, 3), 'int32', 0, (12, 4)),
    ((2, 3), 'int32', 1, (28, 8)),
    ((2, 3), 'int32', False, (12, 4)),
    ((2, 3), 'int32', True, (28, 8)),
    ((2, 3), 'int32', None, (28, 8)),
    ((2, 3), 'int16', (2, 3), (28, 8)),
    ((2, 3, 4), 'int32', (7, 3, 5), (352, 108, 24)),
])
def test_dummy_ndarray_padding(xp, shape, dtype, padding, expected_strides):
    if padding is None:
        a = array_utils.create_dummy_ndarray(xp, shape, dtype)
    else:
        a = array_utils.create_dummy_ndarray(xp, shape, dtype, padding=padding)
    assert isinstance(a, xp.ndarray)
    assert a.shape == shape
    assert a.dtype == xp.dtype(dtype)
    assert a.strides == expected_strides


@pytest.mark.parametrize('shape', [
    (),
    (0,),
    (2, 3),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_check_device(shape, device):
    dtype = 'float32'
    a = chainerx.empty(shape, dtype, device=device)

    array_utils.check_device(a, device.name)
    array_utils.check_device(a, device)


@pytest.mark.parametrize('device_spec', [None, 'native', 'native:0'])
def test_check_device_device_spec(shape, device_spec):
    dtype = 'float32'
    a = chainerx.empty(shape, dtype, device=device_spec)
    device = chainerx.get_device(device_spec)

    array_utils.check_device(a, device_spec)
    array_utils.check_device(a, device)


@pytest.mark.parametrize_device(['native:0'])
@pytest.mark.parametrize('compare_device_spec', [None, 'native:1'])
@pytest.mark.parametrize('shape', [
    (),
    (0,),
    (2, 3),
])
def test_check_device_fail(shape, device, compare_device_spec):
    dtype = 'float32'
    a = chainerx.empty(shape, dtype, device=device)

    with chainerx.device_scope('native:1'):
        with pytest.raises(AssertionError):
            array_utils.check_device(a, compare_device_spec)
