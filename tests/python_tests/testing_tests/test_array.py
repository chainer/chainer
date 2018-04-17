import numpy
import pytest

import xchainer
import xchainer.testing


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


def test_assert_array_equal_fail(dtype):
    a = numpy.zeros((2, 3), dtype)
    b = numpy.zeros((2, 3), dtype)
    a[1, 1] = 0
    b[1, 1] = 2
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


@pytest.mark.xfail
def test_assert_allclose_fail():
    a = numpy.zeros((2, 3), numpy.float32)
    b = numpy.zeros((2, 3), numpy.float32)
    a[1, 1] = 1
    b[1, 1] = 2
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


def test_assert_allclose_fail_equal_nan_disabled():
    with pytest.raises(AssertionError):
        xchainer.testing.assert_allclose(float('nan'), float('nan'), equal_nan=False)
    a = numpy.zeros((2, 3), numpy.float32)
    b = numpy.zeros((2, 3), numpy.float32)
    a[1, 1] = float('nan')
    b[1, 1] = float('nan')
    with pytest.raises(AssertionError):
        xchainer.testing.assert_allclose(a, b, equal_nan=False)
