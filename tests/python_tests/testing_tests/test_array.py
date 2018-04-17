import numpy
import pytest

import xchainer
import xchainer.testing


def _make_onehot_arrays(shape, dtype, value1, value2):
    a = numpy.zeros(shape, dtype)
    b = numpy.zeros(shape, dtype)
    indices = list(numpy.ndindex(*shape))
    a[indices[len(indices) // 2]] = value1
    b[indices[len(indices) // 2]] = value2
    return a, b


@pytest.mark.parametrize('shape', [(), (0,), (1,), (2, 3)])
def test_assert_array_equal(shape, dtype):
    np_a = numpy.arange(2, 2 + numpy.prod(shape)).astype(dtype).reshape(shape)
    xc_a = xchainer.array(np_a)
    xchainer.testing.assert_array_equal(np_a, np_a)
    xchainer.testing.assert_array_equal(xc_a, xc_a)
    xchainer.testing.assert_array_equal(np_a, xc_a)
    xchainer.testing.assert_array_equal(xc_a, np_a)
    xchainer.testing.assert_array_equal(np_a, numpy.array(np_a))
    xchainer.testing.assert_array_equal(xc_a, xchainer.array(xc_a))


@pytest.mark.parametrize('shape', [(), (1,), (2, 3)])
def test_assert_array_equal_fail(shape, dtype):
    a, b = _make_onehot_arrays(shape, dtype, 0, 2)
    with pytest.raises(AssertionError):
        xchainer.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('value1,value2', [
    (True, 1),
    (True, 1.0),
    (False, 0),
    (False, 0.0),
    (2.0, 2),
    (numpy.int32(2), 2.0),
    (xchainer.Scalar(2), 2.0),
    (xchainer.Scalar(2, xchainer.int64), numpy.float32(2)),
    (float('nan'), numpy.float32('nan')),
])
def test_assert_array_equal_scalar(value1, value2):
    xchainer.testing.assert_array_equal(value1, value2)
    xchainer.testing.assert_array_equal(value2, value1)


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
        xchainer.testing.assert_array_equal(value1, value2)
    with pytest.raises(AssertionError):
        xchainer.testing.assert_array_equal(value2, value1)


@pytest.mark.parametrize('shape', [(), (0,), (1,), (2, 3)])
def test_assert_allclose(shape, dtype):
    np_a = numpy.arange(2, 2 + numpy.prod(shape)).astype(dtype).reshape(shape)
    xc_a = xchainer.array(np_a)
    xchainer.testing.assert_allclose(np_a, np_a)
    xchainer.testing.assert_allclose(xc_a, xc_a)
    xchainer.testing.assert_allclose(np_a, xc_a)
    xchainer.testing.assert_allclose(xc_a, np_a)
    xchainer.testing.assert_allclose(np_a, numpy.array(np_a))
    xchainer.testing.assert_allclose(xc_a, xchainer.array(xc_a))


@pytest.mark.parametrize('shape', [(), (1,), (2, 3)])
def test_assert_allclose_fail(shape, dtype):
    a, b = _make_onehot_arrays(shape, dtype, 0, 2)
    with pytest.raises(AssertionError):
        xchainer.testing.assert_allclose(a, b)


@pytest.mark.parametrize('value1,value2', [
    (True, 1),
    (True, 1.0),
    (False, 0),
    (False, 0.0),
    (2.0, 2),
    (numpy.int32(2), 2.0),
    (xchainer.Scalar(2), 2.0),
    (xchainer.Scalar(2, xchainer.int64), numpy.float32(2)),
    (float('nan'), numpy.float32('nan')),
])
def test_assert_allclose_scalar(value1, value2):
    xchainer.testing.assert_allclose(value1, value2)
    xchainer.testing.assert_allclose(value2, value1)


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
        xchainer.testing.assert_allclose(value1, value2)
    with pytest.raises(AssertionError):
        xchainer.testing.assert_allclose(value2, value1)


def test_assert_allclose_fail_equal_nan():
    xchainer.testing.assert_allclose(float('nan'), float('nan'))
    with pytest.raises(AssertionError):
        xchainer.testing.assert_allclose(float('nan'), float('nan'), equal_nan=False)

    a = numpy.zeros((2, 3), numpy.float32)
    b = numpy.zeros((2, 3), numpy.float32)
    a[1, 1] = float('nan')
    b[1, 1] = float('nan')
    xchainer.testing.assert_allclose(a, b)
    with pytest.raises(AssertionError):
        xchainer.testing.assert_allclose(a, b, equal_nan=False)


@pytest.mark.parametrize('shape', [(), (1,), (2, 3)])
def test_assert_allclose_exact(xp, shape, dtype):
    a, b = _make_onehot_arrays(shape, dtype, 1.0, 1.0)
    xchainer.testing.assert_allclose(a, b)


def test_assert_allclose_close(xp, float_dtype):
    dtype = float_dtype
    shape = (2, 3)
    a, b = _make_onehot_arrays(shape, dtype, 1.0, 1.0 + 5e-8)
    xchainer.testing.assert_allclose(a, b)


def test_assert_allclose_fail_not_close(float_dtype):
    dtype = float_dtype
    shape = (2, 3)
    a, b = _make_onehot_arrays(shape, dtype, 1.0, 1.0 + 2e-7)
    with pytest.raises(AssertionError):
        xchainer.testing.assert_allclose(a, b)


def test_assert_allclose_close2(float_dtype):
    dtype = float_dtype
    shape = (2, 3)
    a, b = _make_onehot_arrays(shape, dtype, 1e8, 1e8 + 5)
    xchainer.testing.assert_allclose(a, b)


def test_assert_allclose_fail_not_close2(float_dtype):
    dtype = float_dtype
    shape = (2, 3)
    a, b = _make_onehot_arrays(shape, dtype, 1e8, 1e8 + 20)
    with pytest.raises(AssertionError):
        xchainer.testing.assert_allclose(a, b)


def test_assert_allclose_rtol(float_dtype):
    dtype = float_dtype
    shape = (2, 3)
    a, b = _make_onehot_arrays(shape, dtype, 1e8, 1e8 + 5e5)
    xchainer.testing.assert_allclose(a, b, rtol=1e-2, atol=0)


def test_assert_allclose_fail_rtol(float_dtype):
    dtype = float_dtype
    shape = (2, 3)
    a, b = _make_onehot_arrays(shape, dtype, 1e8, 1e8 + 2e6)
    with pytest.raises(AssertionError):
        xchainer.testing.assert_allclose(a, b, rtol=1e-2, atol=0)


def test_assert_allclose_atol(float_dtype):
    dtype = float_dtype
    shape = (2, 3)
    a, b = _make_onehot_arrays(shape, dtype, 1e8, 1e8 + 5e1)
    xchainer.testing.assert_allclose(a, b, rtol=0, atol=1e2)


def test_assert_allclose_fail_atol(float_dtype):
    dtype = float_dtype
    shape = (2, 3)
    a, b = _make_onehot_arrays(shape, dtype, 1e8, 1e8 + 2e2)
    with pytest.raises(AssertionError):
        xchainer.testing.assert_allclose(a, b, rtol=0, atol=1e2)
