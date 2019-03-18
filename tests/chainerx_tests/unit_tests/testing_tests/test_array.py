import numpy
import pytest

import chainerx
import chainerx.testing


def _make_onehot_arrays(shape, dtype, value1, value2):
    a = numpy.zeros(shape, dtype)
    b = numpy.zeros(shape, dtype)
    indices = list(numpy.ndindex(*shape))
    a[indices[len(indices) // 2]] = value1
    b[indices[len(indices) // 2]] = value2
    return a, b


@pytest.mark.parametrize(
    'dtype1,dtype2',
    list(zip(chainerx.testing.all_dtypes, chainerx.testing.all_dtypes)) + [
        (numpy.float32, numpy.int64),  # arrays with different dtypes
    ])
@pytest.mark.parametrize('shape,transpose', [
    ((), False),
    ((0,), False),
    ((1,), False),
    ((2, 3), False),
    ((2, 3), True),  # arrays with different strides
])
def test_assert_array_equal(shape, transpose, dtype1, dtype2):
    np_a = numpy.arange(2, 2 + numpy.prod(shape)).astype(dtype1).reshape(shape)
    if transpose:
        np_b = numpy.empty(np_a.T.shape, dtype=dtype2).T
        np_b[:] = np_a
    else:
        np_b = numpy.arange(2, 2 + numpy.prod(shape)
                            ).astype(dtype2).reshape(shape)

    chx_a = chainerx.array(np_a)
    chx_b = chainerx.array(np_b)

    # Test precondition checks
    assert np_a.shape == np_b.shape
    if transpose:
        assert np_a.strides != np_b.strides, 'transpose=True is meaningless'

    # Test checks
    chainerx.testing.assert_array_equal(np_a, np_a)  # np-np (same obj)
    chainerx.testing.assert_array_equal(chx_a, chx_a)  # chx-chx (same obj)
    chainerx.testing.assert_array_equal(np_a, np_b)  # np-np (diff. obj)
    chainerx.testing.assert_array_equal(chx_a, chx_b)  # chx-chx (diff. obj)
    chainerx.testing.assert_array_equal(np_a, chx_b)  # np-chx
    chainerx.testing.assert_array_equal(chx_a, np_b)  # chx-np


@pytest.mark.parametrize('shape', [(), (1,), (2, 3)])
def test_assert_array_equal_fail(shape, dtype):
    a, b = _make_onehot_arrays(shape, dtype, 0, 2)
    with pytest.raises(AssertionError):
        chainerx.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('value1,value2', [
    (True, 1),
    (True, 1.0),
    (False, 0),
    (False, 0.0),
    (2.0, 2),
    (numpy.int32(2), 2.0),
    (chainerx.Scalar(2), 2.0),
    (chainerx.Scalar(2), numpy.float32(2)),
    (float('nan'), numpy.float32('nan')),
])
def test_assert_array_equal_scalar(value1, value2):
    chainerx.testing.assert_array_equal(value1, value2)
    chainerx.testing.assert_array_equal(value2, value1)


@pytest.mark.parametrize('value1,value2', [
    (2, 3),
    (2.0, 3),
    (True, 0),
    (True, -1),
    (False, 1),
    (float('nan'), float('inf')),
])
def test_assert_array_equal_fail_scalar(value1, value2):
    with pytest.raises(AssertionError):
        chainerx.testing.assert_array_equal(value1, value2)
    with pytest.raises(AssertionError):
        chainerx.testing.assert_array_equal(value2, value1)


@pytest.mark.parametrize(
    'dtype1,dtype2',
    list(zip(chainerx.testing.all_dtypes, chainerx.testing.all_dtypes)) + [
        (numpy.float32, numpy.int64),  # arrays with different dtypes
    ])
@pytest.mark.parametrize('shape,transpose', [
    ((), False),
    ((0,), False),
    ((1,), False),
    ((2, 3), False),
    ((2, 3), True),  # arrays with different strides
])
def test_assert_allclose(shape, transpose, dtype1, dtype2):
    atol = 1e-3 if numpy.dtype('float16') in [dtype1, dtype2] else 1e-5

    np_a = numpy.arange(2, 2 + numpy.prod(shape)).astype(dtype1).reshape(shape)
    if transpose:
        np_b = numpy.empty(np_a.T.shape, dtype=dtype2).T
        np_b[:] = np_a
    else:
        np_b = numpy.arange(2, 2 + numpy.prod(shape)
                            ).astype(dtype2).reshape(shape)

    # Give some perturbation only if dtype is float
    if np_a.dtype.kind in ('f', 'c'):
        np_a += atol * 1e-1
    if np_b.dtype.kind in ('f', 'c'):
        np_b -= atol * 1e-1

    chx_a = chainerx.array(np_a)
    chx_b = chainerx.array(np_b)

    # Test precondition checks
    assert np_a.shape == np_b.shape
    if transpose:
        assert np_a.strides != np_b.strides, 'transpose=True is meaningless'

    # Test checks
    chainerx.testing.assert_allclose(np_a, np_a, atol=atol)  # np-np (same obj)
    chainerx.testing.assert_allclose(
        chx_a, chx_a, atol=atol)  # chx-chx (same obj)
    chainerx.testing.assert_allclose(
        np_a, np_b, atol=atol)  # np-np (diff. obj)
    chainerx.testing.assert_allclose(
        chx_a, chx_b, atol=atol)  # chx-chx (diff. obj)
    chainerx.testing.assert_allclose(np_a, chx_b, atol=atol)  # np-chx
    chainerx.testing.assert_allclose(chx_a, np_b, atol=atol)  # chx-np


@pytest.mark.parametrize('shape', [(), (1,), (2, 3)])
def test_assert_allclose_fail(shape, dtype):
    a, b = _make_onehot_arrays(shape, dtype, 0, 2)
    with pytest.raises(AssertionError):
        chainerx.testing.assert_allclose(a, b)


@pytest.mark.parametrize('value1,value2', [
    (True, 1),
    (True, 1.0),
    (False, 0),
    (False, 0.0),
    (2.0, 2),
    (numpy.int32(2), 2.0),
    (chainerx.Scalar(2), 2.0),
    (chainerx.Scalar(2), numpy.float32(2)),
    (float('nan'), numpy.float32('nan')),
])
def test_assert_allclose_scalar(value1, value2):
    chainerx.testing.assert_allclose(value1, value2)
    chainerx.testing.assert_allclose(value2, value1)


@pytest.mark.parametrize('value1,value2', [
    (2, 3),
    (2.0, 3),
    (True, 0),
    (True, -1),
    (False, 1),
    (float('nan'), float('inf')),
])
def test_assert_allclose_fail_scalar(value1, value2):
    with pytest.raises(AssertionError):
        chainerx.testing.assert_allclose(value1, value2)
    with pytest.raises(AssertionError):
        chainerx.testing.assert_allclose(value2, value1)


def test_assert_allclose_fail_equal_nan():
    chainerx.testing.assert_allclose(float('nan'), float('nan'))
    with pytest.raises(AssertionError):
        chainerx.testing.assert_allclose(
            float('nan'), float('nan'), equal_nan=False)

    shape = (2, 3)
    dtype = numpy.float32
    a, b = _make_onehot_arrays(shape, dtype, float('nan'), float('nan'))
    chainerx.testing.assert_allclose(a, b)
    with pytest.raises(AssertionError):
        chainerx.testing.assert_allclose(a, b, equal_nan=False)


@pytest.mark.parametrize('shape', [(), (1,), (2, 3)])
def test_assert_allclose_exact(shape, dtype):
    a, b = _make_onehot_arrays(shape, dtype, 1.0, 1.0)
    chainerx.testing.assert_allclose(a, b)


def test_assert_allclose_close_default_tol():
    dtype = numpy.float64
    shape = (2, 3)

    # small absolute error
    a, b = _make_onehot_arrays(shape, dtype, 1.0, 1.0 + 5e-8)
    chainerx.testing.assert_allclose(a, b)

    # large absolute error
    a, b = _make_onehot_arrays(shape, dtype, 1e8, 1e8 + 5)
    chainerx.testing.assert_allclose(a, b)

    # expected failure: small absolute error
    a, b = _make_onehot_arrays(shape, dtype, 1.0, 1.0 + 2e-7)
    with pytest.raises(AssertionError):
        chainerx.testing.assert_allclose(a, b)

    # expected failure: large absolute error
    a, b = _make_onehot_arrays(shape, dtype, 1e8, 1e8 + 20)
    with pytest.raises(AssertionError):
        chainerx.testing.assert_allclose(a, b)


def test_assert_allclose_rtol(float_dtype):
    dtype = float_dtype
    shape = (2, 3)

    # relative error < rtol
    a, b = _make_onehot_arrays(shape, dtype, 1e4, 1e4 + 50)
    chainerx.testing.assert_allclose(a, b, rtol=1e-2, atol=0)

    # relative error > rtol
    a, b = _make_onehot_arrays(shape, dtype, 1e4, 1e4 + 200)
    with pytest.raises(AssertionError):
        chainerx.testing.assert_allclose(a, b, rtol=1e-2, atol=0)


def test_assert_allclose_atol(float_dtype):
    dtype = float_dtype
    shape = (2, 3)

    # absolute error < atol
    a, b = _make_onehot_arrays(shape, dtype, 1e-3, 1e-3 + 1e-2)
    chainerx.testing.assert_allclose(a, b, rtol=0, atol=2e-2)

    # absolute error > atol
    a, b = _make_onehot_arrays(shape, dtype, 1e-3, 1e-3 + 1e-2)
    with pytest.raises(AssertionError):
        chainerx.testing.assert_allclose(a, b, rtol=0, atol=5e-3)


def test_assert_array_equal_ex_fail_dtype():
    shape = (3, 2)
    dtype1 = numpy.float32
    dtype2 = numpy.int64
    a = numpy.arange(2, 2 + numpy.prod(shape)).astype(dtype1).reshape(shape)
    b = a.astype(dtype2)
    with pytest.raises(AssertionError):
        chainerx.testing.assert_array_equal_ex(a, b)
    with pytest.raises(AssertionError):
        # strides_check does not affect dtype_check
        chainerx.testing.assert_array_equal_ex(a, b, strides_check=False)
    chainerx.testing.assert_array_equal_ex(a, b, dtype_check=False)


def test_assert_array_equal_ex_fail_strides():
    shape = (3, 2)
    dtype = numpy.float32
    a = numpy.arange(2, 2 + numpy.prod(shape)).astype(dtype).reshape(shape)
    b = numpy.empty(a.T.shape, dtype).T
    b[:] = a
    with pytest.raises(AssertionError):
        chainerx.testing.assert_array_equal_ex(a, b)
    chainerx.testing.assert_array_equal_ex(a, b, strides_check=False)
    # dtype_check=False implies strides_check=False
    chainerx.testing.assert_array_equal_ex(a, b, dtype_check=False)
