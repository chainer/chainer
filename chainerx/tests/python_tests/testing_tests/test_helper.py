import numpy
import pytest

import xchainer
import xchainer.testing


class FooError(Exception):
    pass


class BarError(Exception):
    pass


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('np_result, xc_result', [
    (1.0, 1.0),
    (numpy.full((1,), 1.0, numpy.float32), xchainer.full((1,), 1.0, xchainer.float32)),
])
def test_numpy_xchainer_array_equal_both_return_nonarray(xp, np_result, xc_result):
    if xp is numpy:
        return np_result
    else:
        return xc_result


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('np_result, xc_result', [
    (None, None),  # Both return None
    (None, xchainer.full((1,), 1.0, xchainer.float32)),  # NumPy returns None
    (numpy.full((1,), 1.0, numpy.float32), None),  # Xchainer returns None
    (numpy.full((1,), 1.0, numpy.float32), xchainer.full((1,), 2.0, xchainer.float32)),  # Value mismatch
    (1.0, xchainer.full((1,), 1.0, xchainer.float64)),  # NumPy returns non-array
    (numpy.full((1,), 1.0, numpy.float64), 1.0),  # Xchainer returns non-array
    (1.0, 1),  # Scalar type mismatch
    (numpy.int64(1), numpy.int64(1)),  # Xchainer returns NumPy scalar
    (numpy.full((1,), 1.0, numpy.float64), numpy.full((1,), 1.0, numpy.float64)),  # Both return NumPy array
    (xchainer.full((1,), 1.0, xchainer.float64), xchainer.full((1,), 1.0, xchainer.float64)),  # Both return Xchainer array
    (xchainer.full((1,), 1.0, xchainer.float64), numpy.full((1,), 1.0, numpy.float64)),  # Return arrays wrong way around
    (numpy.full((1,), 1.0, numpy.float64), xchainer.full((1,), 1.0, xchainer.float32)),  # Dtype mismatch
    (numpy.full((1,), 1.0, numpy.float64), xchainer.full((), 1.0, xchainer.float32)),  # Shape mismatch
    (numpy.array([[0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]], numpy.float32)[0:2, 1:3],
     xchainer.array(numpy.array([[0, 0, 0], [1, 2, 0], [3, 4, 0]], numpy.float32))[1:3, 0:2]),  # Strides mismatch
])
def test_numpy_xchainer_array_equal_fail_invalid_return(xp, np_result, xc_result):
    if xp is numpy:
        return np_result
    else:
        return xc_result


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_array_equal()
def test_numpy_xchainer_array_equal_fail_both_raise(xp):
    if xp is numpy:
        raise TypeError('NumPy error')
    else:
        raise TypeError('Xchainer error')


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_array_equal()
def test_numpy_xchainer_array_equal_fail_numpy_raise(xp):
    if xp is numpy:
        raise TypeError('NumPy error')
    else:
        return xchainer.full((1,), 1.0, xchainer.float32)


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_array_equal()
def test_numpy_xchainer_array_equal_fail_xchainer_raise(xp):
    if xp is numpy:
        return numpy.full((1,), 1.0, numpy.float32)
    else:
        raise TypeError('Xchainer error')


@xchainer.testing.numpy_xchainer_array_equal()
def test_numpy_xchainer_array_equal_parametrize_dtype(xp, dtype):
    assert isinstance(dtype, str)
    assert dtype in xchainer.testing.all_dtypes
    if xp is numpy:
        return numpy.full((1,), 1.0, dtype)
    else:
        return xchainer.full((1,), 1.0, dtype)


@xchainer.testing.numpy_xchainer_array_equal(dtype_check=False)
def test_numpy_xchainer_array_equal_dtype_check_disabled(xp):
    if xp is numpy:
        return numpy.full((1,), 1.0, numpy.float64)
    else:
        return xchainer.full((1,), 1.0, xchainer.float32)


@xchainer.testing.numpy_xchainer_array_equal(name='foo')
def test_numpy_xchainer_array_equal_name(foo):
    if foo is numpy:
        return numpy.full((1,), 1.0, numpy.float32)
    else:
        return xchainer.full((1,), 1.0, xchainer.float32)


@xchainer.testing.numpy_xchainer_array_equal(accept_error=FooError)
def test_numpy_xchainer_array_equal_accept_error(xp):
    raise FooError()


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_array_equal(accept_error=FooError)
def test_numpy_xchainer_array_equal_fail_accept_error_differ(xp):
    raise BarError()


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_array_equal(accept_error=FooError)
def test_numpy_xchainer_array_equal_fail_accept_error_only_numpy(xp):
    if xp is numpy:
        raise FooError()
    else:
        return xchainer.full((1,), 1.0, xchainer.float32)


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_array_equal(accept_error=FooError)
def test_numpy_xchainer_array_equal_fail_accept_error_only_xchainer(xp):
    if xp is numpy:
        return numpy.full((1,), 1.0, numpy.float32)
    else:
        raise FooError()


@xchainer.testing.numpy_xchainer_array_equal(accept_error=(FooError, BarError))
def test_numpy_xchainer_array_equal_accept_error_multiple(xp):
    if xp is numpy:
        raise FooError()
    else:
        raise BarError()


@xchainer.testing.numpy_xchainer_array_equal()
def test_numpy_xchainer_array_equal_nan(xp):
    a = numpy.zeros((5, 3), numpy.float32)
    if xp is numpy:
        a[2, 1] = float('nan')
    else:
        a[2, 1] = float('nan')
    return xp.array(a)


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_array_equal()
def test_numpy_xchainer_array_equal_fail_nan_inf(xp):
    a = numpy.zeros((5, 3), numpy.float32)
    if xp is numpy:
        a[2, 1] = float('nan')
    else:
        a[2, 1] = float('inf')
    return xp.array(a)


@xchainer.testing.numpy_xchainer_array_equal()
def test_numpy_xchainer_array_equal_accept_ignore(xp):
    return xchainer.testing.ignore()


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_array_equal()
def test_numpy_xchainer_array_equal_fail_numpy_ignore(xp):
    if xp is numpy:
        return xchainer.testing.ignore()
    else:
        return xp.arange(10)


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_array_equal()
def test_numpy_xchainer_array_equal_fail_xchainer_ignore(xp):
    if xp is numpy:
        return xp.arange(10)
    else:
        return xchainer.testing.ignore()


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_array_equal()
def test_numpy_xchainer_array_equal_fail_ignore_none(xp):
    if xp is numpy:
        return xchainer.testing.ignore()
    else:
        return None


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_array_equal()
def test_numpy_xchainer_array_equal_fail_ignore_error(xp):
    if xp is numpy:
        return xchainer.testing.ignore()
    else:
        raise FooError()


@xchainer.testing.numpy_xchainer_allclose()
def test_numpy_xchainer_allclose_exact(xp):
    assert xp is numpy or xp is xchainer
    a = numpy.zeros((5, 3), numpy.float32)
    if xp is numpy:
        a[2, 1] = 1.0
    else:
        a[2, 1] = 1.0
    return xp.array(a)


@xchainer.testing.numpy_xchainer_allclose()
def test_numpy_xchainer_allclose_close(xp):
    a = numpy.zeros((5, 3), numpy.float32)
    if xp is numpy:
        a[2, 1] = 1.0
    else:
        a[2, 1] = 1.0 + 5e-8
    return xp.array(a)


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose()
def test_numpy_xchainer_allclose_fail_not_close(xp):
    a = numpy.zeros((5, 3), numpy.float32)
    if xp is numpy:
        a[2, 1] = 1.0
    else:
        a[2, 1] = 1.0 + 2e-7
    return xp.array(a)


@xchainer.testing.numpy_xchainer_allclose()
def test_numpy_xchainer_allclose_close2(xp):
    a = numpy.zeros((5, 3), numpy.float32)
    if xp is numpy:
        a[2, 1] = 1e8
    else:
        a[2, 1] = 1e8 + 5
    return xp.array(a)


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose()
def test_numpy_xchainer_allclose_fail_not_close2(xp):
    a = numpy.zeros((5, 3), numpy.float32)
    if xp is numpy:
        a[2, 1] = 1e8
    else:
        a[2, 1] = 1e8 + 20
    return xp.array(a)


@xchainer.testing.numpy_xchainer_allclose(rtol=1e-2, atol=0)
def test_numpy_xchainer_allclose_rtol(xp):
    a = numpy.zeros((5, 3), numpy.float32)
    if xp is numpy:
        a[2, 1] = 1e8
    else:
        a[2, 1] = 1e8 + 5e5
    return xp.array(a)


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose(rtol=1e-2, atol=0)
def test_numpy_xchainer_allclose_fail_rtol(xp):
    a = numpy.zeros((5, 3), numpy.float32)
    if xp is numpy:
        a[2, 1] = 1e8
    else:
        a[2, 1] = 1e8 + 2e6
    return xp.array(a)


@xchainer.testing.numpy_xchainer_allclose(rtol=0, atol=1e2)
def test_numpy_xchainer_allclose_atol(xp):
    a = numpy.zeros((5, 3), numpy.float32)
    if xp is numpy:
        a[2, 1] = 1e8
    else:
        a[2, 1] = 1e8 + 5e1
    return xp.array(a)


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose(rtol=0, atol=1e2)
def test_numpy_xchainer_allclose_fail_atol(xp):
    a = numpy.zeros((5, 3), numpy.float32)
    if xp is numpy:
        a[2, 1] = 1e8
    else:
        a[2, 1] = 1e8 + 2e2
    return xp.array(a)


@xchainer.testing.numpy_xchainer_allclose()
def test_numpy_xchainer_allclose_nan(xp):
    a = numpy.zeros((5, 3), numpy.float32)
    if xp is numpy:
        a[2, 1] = float('nan')
    else:
        a[2, 1] = float('nan')
    return xp.array(a)


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose(equal_nan=False)
def test_numpy_xchainer_allclose_fail_nan_disabled(xp):
    a = numpy.zeros((5, 3), numpy.float32)
    if xp is numpy:
        a[2, 1] = float('nan')
    else:
        a[2, 1] = float('nan')
    return xp.array(a)


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose()
def test_numpy_xchainer_allclose_fail_nan_inf(xp):
    a = numpy.zeros((5, 3), numpy.float32)
    if xp is numpy:
        a[2, 1] = float('nan')
    else:
        a[2, 1] = float('inf')
    return xp.array(a)


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose()
def test_numpy_xchainer_allclose_fail_both_numpy(xp):
    if xp is numpy:
        return numpy.full((1,), 1.0, numpy.float64)
    else:
        return numpy.full((1,), 1.0, numpy.float64)


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose()
def test_numpy_xchainer_allclose_fail_both_xchainer(xp):
    if xp is numpy:
        return xchainer.full((1,), 1.0, xchainer.float64)
    else:
        return xchainer.full((1,), 1.0, xchainer.float64)


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose()
def test_numpy_xchainer_allclose_fail_wrong_way_around(xp):
    if xp is numpy:
        return xchainer.full((1,), 1.0, xchainer.float64)
    else:
        return numpy.full((1,), 1.0, numpy.float64)


@xchainer.testing.numpy_xchainer_allclose(name='foo')
def test_numpy_xchainer_allclose_name(foo):
    if foo is numpy:
        return numpy.full((1,), 1.0, numpy.float64)
    else:
        return xchainer.full((1,), 1.0, xchainer.float64)


@xchainer.testing.numpy_xchainer_allclose(accept_error=FooError)
def test_numpy_xchainer_allclose_accept_error(xp):
    raise FooError()


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose(accept_error=FooError)
def test_numpy_xchainer_allclose_fail_accept_error_differ(xp):
    raise BarError()


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose(accept_error=FooError)
def test_numpy_xchainer_allclose_fail_accept_error_only_numpy(xp):
    if xp is numpy:
        raise FooError()
    else:
        return xchainer.full((1,), 1.0, xchainer.float32)


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose(accept_error=FooError)
def test_numpy_xchainer_allclose_fail_accept_error_only_xchainer(xp):
    if xp is numpy:
        return numpy.full((1,), 1.0, numpy.float32)
    else:
        raise FooError()


@xchainer.testing.numpy_xchainer_allclose(accept_error=(FooError, BarError))
def test_numpy_xchainer_allclose_accept_error_multiple(xp):
    if xp is numpy:
        raise FooError()
    else:
        raise BarError()


@xchainer.testing.numpy_xchainer_allclose()
def test_numpy_xchainer_allclose_accept_ignore(xp):
    return xchainer.testing.ignore()


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose()
def test_numpy_xchainer_allclose_fail_numpy_ignore(xp):
    if xp is numpy:
        return xchainer.testing.ignore()
    else:
        return xp.arange(10)


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose()
def test_numpy_xchainer_allclose_fail_xchainer_ignore(xp):
    if xp is numpy:
        return xp.arange(10)
    else:
        return xchainer.testing.ignore()


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose()
def test_numpy_xchainer_allclose_fail_ignore_none(xp):
    if xp is numpy:
        return xchainer.testing.ignore()
    else:
        return None


@pytest.mark.xfail(strict=True)
@xchainer.testing.numpy_xchainer_allclose()
def test_numpy_xchainer_allclose_fail_ignore_error(xp):
    if xp is numpy:
        return xchainer.testing.ignore()
    else:
        raise FooError()
