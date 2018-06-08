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


class _ResultsCheckFailure(Exception):
    def __init__(self, msg, indices, condense_results_func=None):
        self.msg = msg
        self.indices = tuple(indices)
        if condense_results_func is None:
            def condense_results_func(np_r, xc_r):
                return f'xchainer: {xc_r} numpy: {np_r}'
        self.condense_results_func = condense_results_func

    def condense_results(self, numpy_result, xchainer_result):
        # Generates a condensed error message for a pair of lowest-level numpy and xchainer results.
        return self.condense_results_func(numpy_result, xchainer_result)


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


def _is_numpy_type(result):
    return isinstance(result, (numpy.ndarray, numpy.generic))


def _check_xchainer_numpy_result_array(check_result_func, xchainer_result, numpy_result, indices):
    # Compares `xchainer_result` and `numpy_result` as arrays.

    is_xchainer_valid_type = isinstance(xchainer_result, xchainer.ndarray)
    is_numpy_valid_type = _is_numpy_type(numpy_result)

    if not (is_xchainer_valid_type and is_numpy_valid_type):
        raise _ResultsCheckFailure(
            'Using decorator without returning ndarrays. '
            'If you want to explicitly ignore certain tests, '
            'return xchainer.testing.ignore() to avoid this error', indices)

    if xchainer_result.shape != numpy_result.shape:
        raise _ResultsCheckFailure(
            'Shape mismatch', indices,
            lambda np_r, xc_r: f'xchainer: {xc_r.shape}, numpy: {np_r.shape}')

    if xchainer_result.device is not xchainer.get_default_device():
        raise _ResultsCheckFailure(
            'Xchainer bad device', indices,
            lambda np_r, xc_r: f'default: {xchainer.get_default_device()}, xchainer: {xc_r.device}')

    try:
        check_result_func(xchainer_result, numpy_result)
    except AssertionError as e:
        # Convert AssertionError to _ResultsCheckFailure
        raise _ResultsCheckFailure(str(e), indices)


def _check_xchainer_numpy_result_impl(check_result_func, xchainer_result, numpy_result, indices):
    # This function raises _ResultsCheckFailure if any failure occurs.
    # `indices` is a tuple of indices to reach both `xchainer_results` and `numpy_results` from top-level results.

    if xchainer_result is _ignored_result and numpy_result is _ignored_result:
        return

    if isinstance(xchainer_result, tuple):
        if not isinstance(numpy_result, tuple):
            raise _ResultsCheckFailure('Different result types', indices)
        if len(xchainer_result) != len(numpy_result):
            raise _ResultsCheckFailure('Result length mismatch', indices)
        for i, (xc_r, np_r) in enumerate(zip(xchainer_result, numpy_result)):
            _check_xchainer_numpy_result_impl(check_result_func, xc_r, np_r, indices + (i,))

    elif isinstance(xchainer_result, xchainer.ndarray):
        _check_xchainer_numpy_result_array(check_result_func, xchainer_result, numpy_result, indices)

    else:
        if _is_numpy_type(xchainer_result):
            raise _ResultsCheckFailure('xchainer result should not be a NumPy type', indices)
        if type(xchainer_result) != type(numpy_result):
            raise _ResultsCheckFailure('Type mismatch', indices)
        if xchainer_result != numpy_result:
            raise _ResultsCheckFailure('Not equal', indices)


def _check_xchainer_numpy_result(check_result_func, xchainer_result, numpy_result):
    # Catch _ResultsCheckFailure and generate a comprehensible error message.
    try:
        _check_xchainer_numpy_result_impl(check_result_func, xchainer_result, numpy_result, indices=())
    except _ResultsCheckFailure as e:
        indices = e.indices
        xc_r = xchainer_result
        np_r = numpy_result
        i = 0
        while len(indices[i:]) > 0:
            xc_r = xc_r[indices[i]]
            np_r = np_r[indices[i]]
            i += 1

        def make_message(e):
            indices_str = ''.join(f'[{i}]' for i in indices)
            s = f'{e.msg}: {e.condense_results(np_r, xc_r)}\n\n'
            if len(indices) > 0:
                s += f'xchainer results{indices_str}: {type(xc_r)}\n'
                s += f'{xc_r}\n\n'
                s += f'numpy results{indices_str}: {type(np_r)}\n'
                s += f'{np_r}\n\n'
            s += f'xchainer results: {type(xchainer_result)}\n'
            s += f'{xchainer_result}\n\n'
            s += f'numpy results: {type(numpy_result)}\n'
            s += f'{numpy_result}\n\n'
            return s

        raise AssertionError(make_message(e))


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
            assert xchainer_result is not None and numpy_result is not None, (
                f'Either or both of Xchainer and numpy returned None. xchainer: {xchainer_result}, numpy: {numpy_result}')
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
    def check_result_func(x, y):
        array.assert_array_equal_ex(x, y, err_msg, verbose, dtype_check=dtype_check, strides_check=strides_check)

    return _make_decorator(check_result_func, name, accept_error)
