import functools
import traceback
import warnings

import numpy
import pytest

import xchainer
from xchainer.testing import array


# A test returning this object will have its return value ignored.
#
# This is e.g. useful when a combination of parametrizations and operations unintentionally cover non-supported function calls.
# For instance, you might parametrize over shapes (tuples) which are unpacked and passed to a function.
# While you might want to test empty tuples for module functions, they should maybe be ignored for ndarray functions.
#
# If either xchainer or numpy returns this object, the other module should too.
# Otherwise, the test will be considered inconsistent and be treated as a failure.
_ignored_result = object()


# A wrapper to obtain the ignore object.
def ignore():
    return _ignored_result


def _call_func(impl, args, kw):
    try:
        result = impl(*args, **kw)
        error = None
        tb = None
    except Exception as e:
        result = None
        error = e
        tb = traceback.format_exc()

    return result, error, tb


def _check_xchainer_numpy_error(xchainer_error, xchainer_tb, numpy_error,
                                numpy_tb, accept_error=()):
    # TODO(sonots): Change error class names of xChainer to be similar with NumPy, and check names.
    if xchainer_error is None and numpy_error is None:
        pytest.fail('Both xchainer and numpy are expected to raise errors, but not')
    elif xchainer_error is None:
        pytest.fail('Only numpy raises error\n\n' + numpy_tb)
    elif numpy_error is None:
        pytest.fail('Only xchainer raises error\n\n' + xchainer_tb)
    elif not (isinstance(xchainer_error, accept_error) and
              isinstance(numpy_error, accept_error)):
        msg = '''Both xchainer and numpy raise exceptions

xchainer
%s
numpy
%s
''' % (xchainer_tb, numpy_tb)
        pytest.fail(msg)


def _check_xchainer_numpy_result(check_result_func, xchainer_result, numpy_result):
    is_xchainer_ignored = xchainer_result is _ignored_result
    is_numpy_ignored = numpy_result is _ignored_result

    if is_xchainer_ignored and is_numpy_ignored:
        return  # Ignore without failing.

    assert is_xchainer_ignored is is_numpy_ignored, (
        f'Ignore value mismatch. xchainer: {is_xchainer_ignored}, numpy: {is_numpy_ignored}.')

    is_xchainer_valid_type = isinstance(xchainer_result, xchainer.ndarray)
    is_numpy_valid_type = isinstance(numpy_result, numpy.ndarray) or numpy.isscalar(numpy_result)

    assert is_xchainer_valid_type and is_numpy_valid_type, (
        'Using decorator without returning ndarrays. If you want to explicitly ignore certain tests, '
        f'return xchainer.testing.ignore() to avoid this error: xchainer: {xchainer_result}, numpy: {numpy_result}')

    assert xchainer_result.shape == numpy_result.shape, (
        f'Shape mismatch: xchainer: {xchainer_result.shape}, numpy: {numpy_result.shape}')

    assert xchainer_result.device is xchainer.get_default_device(), (
        f'Xchainer bad device: default: {xchainer.get_default_device()}, xchainer: {xchainer_result.device}')

    check_result_func(xchainer_result, numpy_result)


def _make_decorator(check_result_func, name, accept_error):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(*args, **kw):
            kw[name] = xchainer
            xchainer_result, xchainer_error, xchainer_tb = _call_func(impl, args, kw)

            kw[name] = numpy
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                numpy_result, numpy_error, numpy_tb = _call_func(impl, args, kw)

            if xchainer_error or numpy_error:
                _check_xchainer_numpy_error(xchainer_error, xchainer_tb,
                                            numpy_error, numpy_tb,
                                            accept_error=accept_error)
                return
            _check_xchainer_numpy_result(check_result_func, xchainer_result, numpy_result)
        # Apply dummy parametrization on `name` (e.g. 'xp') to avoid pytest error when collecting test functions.
        return pytest.mark.parametrize(name, [None])(test_func)
    return decorator


def numpy_xchainer_allclose(
        *, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True, name='xp', dtype_check=None, strides_check=None, accept_error=()):
    """numpy_xchainer_allclose(*, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True, name='xp', dtype_check=True, strides_check=True, accept_error=())

    Decorator that checks that NumPy and xChainer results are equal up to a tolerance.

    Args:
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         equal_nan(bool): Allow NaN values if True. Otherwise, fail the assertion if any NaN is found.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either ``numpy`` or ``xchainer`` module.
         dtype_check(bool): If ``True``, consistency of dtype is also checked.
             Disabling ``dtype_check`` also implies ``strides_check=False``.
         strides_check(bool): If ``True``, consistency of strides is also checked.
         accept_error(Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and xChainer test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`numpy_xchainer_allclose`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``xchainer``.

    .. seealso:: :func:`xchainer.testing.assert_allclose_ex`
    """  # NOQA
    if not dtype_check:
        strides_check = False

    def check_result_func(x, y):
        array.assert_allclose_ex(x, y, rtol, atol, equal_nan, err_msg, verbose, dtype_check=dtype_check, strides_check=strides_check)

    return _make_decorator(check_result_func, name, accept_error)


def numpy_xchainer_array_equal(*, err_msg='', verbose=True, name='xp', dtype_check=None, strides_check=None, accept_error=()):
    """numpy_xchainer_array_equal(*, err_msg='', verbose=True, name='xp', dtype_check=True, strides_check=True, accept_error=()):

    Decorator that checks that NumPy and xChainer results are equal.

    Args:
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either ``numpy`` or ``xchainer`` module.
         dtype_check(bool): If ``True``, consistency of dtype is also checked.
             Disabling ``dtype_check`` also implies ``strides_check=False``
         strides_check(bool): If ``True``, consistency of strides is also checked.
         accept_error(Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and xChainer test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`numpy_xchainer_array_equal`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``xchainer``.

    .. seealso:: :func:`xchainer.testing.assert_array_equal_ex`
    """
    if not dtype_check:
        strides_check = False

    def check_result_func(x, y):
        array.assert_array_equal_ex(x, y, err_msg, verbose, dtype_check=dtype_check, strides_check=strides_check)

    return _make_decorator(check_result_func, name, accept_error)
