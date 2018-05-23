import numpy
import pytest

import xchainer

from tests import array_utils


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


@pytest.mark.parametrize('device_spec', [
    'numpy',  # Special value representing xp=numpy
    None, 'native', 'native:0', 'cuda:0',
])
@pytest.mark.parametrize('shape', [
    (),
    (0,),
    (1,),
    (2, 3),
    (1, 1, 1),
    (2, 0, 3),
])
@pytest.mark.parametrize('dtype', xchainer.testing.all_dtypes)
@pytest.mark.parametrize('pattern', [1, 2])
@pytest.mark.parametrize('padding', [True, False])
def test_dummy_ndarray(device_spec, shape, dtype, pattern, padding):
    if device_spec == 'numpy':
        xp = numpy
    else:
        xp = xchainer
    a = array_utils.create_dummy_ndarray(xp, shape, dtype, device=device_spec, pattern=pattern, padding=padding)

    assert isinstance(a, xp.ndarray)
    assert a.dtype == xp.dtype(dtype)
    assert a.shape == shape

    # Check values
    if xp is xchainer:
        a_np = xchainer.tonumpy(a)
    else:
        a_np = a
    if pattern == 1:
        if a.dtype.name == 'bool':
            expected_data = [i % 2 == 1 for i in range(a.size)]
        elif a.dtype.name in xchainer.testing.unsigned_dtypes:
            expected_data = list(range(a.size))
        else:
            expected_data = list(range(-1, a.size - 1))
    else:
        if a.dtype.name == 'bool':
            expected_data = [i % 3 == 0 for i in range(a.size)]
        elif a.dtype.name in xchainer.testing.unsigned_dtypes:
            expected_data = list(range(1, a.size + 1))
        else:
            expected_data = list(range(-2, a.size - 2))
    numpy.testing.assert_equal(a_np.ravel(), expected_data)

    # Check strides
    if xp is xchainer:
        assert a.device is xchainer.get_device(device_spec)
    if not padding:
        if xp is xchainer:
            assert a.is_contiguous
        else:
            assert a.flags.c_contiguous


@pytest.mark.parametrize('xp,shape,dtype,padding,expected_strides', [
    # padding=None means unspecified.
    (numpy, (), 'bool_', (), ()),
    (numpy, (), 'int32', (), ()),
    (numpy, (), 'int32', 1, ()),
    (numpy, (2,), 'bool_', (0,), (1,)),
    (numpy, (2,), 'bool_', (1,), (2,)),
    (numpy, (2,), 'bool_', (2,), (3,)),
    (numpy, (2,), 'bool_', 0, (1,)),
    (numpy, (2,), 'bool_', 1, (2,)),
    (numpy, (2,), 'int32', (0,), (4,)),
    (numpy, (2,), 'int32', (1,), (8,)),
    (numpy, (2,), 'int32', (2,), (12,)),
    (numpy, (2,), 'int32', 0, (4,)),
    (numpy, (2,), 'int32', 1, (8,)),
    (numpy, (0,), 'int32', (0,), (4,)),
    (numpy, (0,), 'int32', (1,), (8,)),
    (numpy, (0,), 'int32', (2,), (12,)),
    (numpy, (2, 3), 'bool_', (0, 0), (3, 1)),
    (numpy, (2, 3), 'bool_', (0, 1), (6, 2)),
    (numpy, (2, 3), 'bool_', (1, 1), (7, 2)),
    (numpy, (2, 3), 'bool_', (2, 3), (14, 4)),
    (numpy, (2, 3), 'bool_', 0, (3, 1)),
    (numpy, (2, 3), 'bool_', 1, (7, 2)),
    (numpy, (2, 3), 'int32', (0, 0,), (12, 4)),
    (numpy, (2, 3), 'int32', (0, 1,), (24, 8)),
    (numpy, (2, 3), 'int32', (1, 1,), (28, 8)),
    (numpy, (2, 3), 'int32', (2, 3,), (56, 16)),
    (numpy, (2, 3), 'int32', 0, (12, 4)),
    (numpy, (2, 3), 'int32', 1, (28, 8)),
    (numpy, (2, 3), 'int32', None, (28, 8)),
    (numpy, (2, 3), 'int16', (2, 3), (28, 8)),
    (xchainer, (2, 3), 'int32', (2, 3), (56, 16)),
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


@pytest.mark.parametrize('device_spec', [
    None, 'native', 'native:0', 'cuda:0',
])
@pytest.mark.parametrize('shape', [
    (),
    (0,),
    (2, 3),
])
def test_check_device(shape, device_spec):
    dtype = 'float32'
    a = xchainer.empty(shape, dtype, device=device_spec)
    device = xchainer.get_device(device_spec)

    array_utils.check_device(a, device_spec)
    array_utils.check_device(a, device)


@pytest.mark.parametrize('device_spec1,device_spec2', [
    ('native', 'cuda'),
    ('native:0', 'native:1'),
])
@pytest.mark.parametrize('shape', [
    (),
    (0,),
    (2, 3),
])
def test_check_device_fail(shape, device_spec1, device_spec2):
    dtype = 'float32'
    a = xchainer.empty(shape, dtype, device=device_spec1)

    with pytest.raises(AssertionError):
        array_utils.check_device(a, device_spec2)
