import functools
import traceback
import warnings

import numpy
import pytest

import chainerx
from chainerx.testing import array


# A test returning this object will have its return value ignored.
#
# This is e.g. useful when a combination of parametrizations and operations
# unintentionally cover non-supported function calls.
# For instance, you might parametrize over shapes (tuples) which are unpacked
# and passed to a function.
# While you might want to test empty tuples for module functions, they should
# maybe be ignored for ndarray functions.
#
# If either chainerx or numpy returns this object, the other module should too.
# Otherwise, the test will be considered inconsistent and be treated as a
# failure.
_ignored_result = object()


# A wrapper to obtain the ignore object.
def ignore():
    return _ignored_result


class _ResultsCheckFailure(Exception):
    def __init__(self, msg, indices, condense_results_func=None):
        self.msg = msg
        self.indices = tuple(indices)
        if condense_results_func is None:
            def condense_results_func(np_r, chx_r):
                return 'chainerx: {} numpy: {}'.format(chx_r, np_r)
        self.condense_results_func = condense_results_func

    def condense_results(self, numpy_result, chainerx_result):
        # Generates a condensed error message for a pair of lowest-level numpy
        # and chainerx results.
        return self.condense_results_func(numpy_result, chainerx_result)


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


def _check_chainerx_numpy_error(chainerx_error, chainerx_tb, numpy_error,
                                numpy_tb, accept_error=()):
    # TODO(sonots): Change error class names of ChainerX to be similar with
    # NumPy, and check names.
    if chainerx_error is None and numpy_error is None:
        pytest.fail(
            'Both chainerx and numpy are expected to raise errors, but not')
    elif chainerx_error is None:
        pytest.fail('Only numpy raises error\n\n' + numpy_tb)
    elif numpy_error is None:
        pytest.fail('Only chainerx raises error\n\n' + chainerx_tb)
    elif not (isinstance(chainerx_error, accept_error) and
              isinstance(numpy_error, accept_error)):
        msg = '''Both chainerx and numpy raise exceptions

chainerx
%s
numpy
%s
''' % (chainerx_tb, numpy_tb)
        pytest.fail(msg)


def _is_numpy_type(result):
    return isinstance(result, (numpy.ndarray, numpy.generic))


def _check_chainerx_numpy_result_array(
        check_result_func, chainerx_result, numpy_result, indices):
    # Compares `chainerx_result` and `numpy_result` as arrays.

    is_chainerx_valid_type = isinstance(chainerx_result, chainerx.ndarray)
    is_numpy_valid_type = _is_numpy_type(numpy_result)

    if not (is_chainerx_valid_type and is_numpy_valid_type):
        raise _ResultsCheckFailure(
            'Using decorator without returning ndarrays. '
            'If you want to explicitly ignore certain tests, '
            'return chainerx.testing.ignore() to avoid this error', indices)

    if chainerx_result.shape != numpy_result.shape:
        raise _ResultsCheckFailure(
            'Shape mismatch', indices,
            lambda np_r, chx_r: (
                'chainerx: {}, numpy: {}'.format(chx_r.shape, np_r.shape)))

    if chainerx_result.device is not chainerx.get_default_device():
        raise _ResultsCheckFailure(
            'ChainerX bad device', indices,
            lambda np_r, chx_r: (
                'default: {}, chainerx: {}'.format(
                    chainerx.get_default_device(), chx_r.device)))

    try:
        check_result_func(chainerx_result, numpy_result)
    except AssertionError as e:
        # Convert AssertionError to _ResultsCheckFailure
        raise _ResultsCheckFailure(str(e), indices)


def _check_chainerx_numpy_result_impl(
        check_result_func, chainerx_result, numpy_result, indices):
    # This function raises _ResultsCheckFailure if any failure occurs.
    # `indices` is a tuple of indices to reach both `chainerx_results` and
    # `numpy_results` from top-level results.

    if chainerx_result is _ignored_result and numpy_result is _ignored_result:
        return

    if isinstance(chainerx_result, (list, tuple)):
        if type(chainerx_result) is not type(numpy_result):
            raise _ResultsCheckFailure('Different result types', indices)
        if len(chainerx_result) != len(numpy_result):
            raise _ResultsCheckFailure('Result length mismatch', indices)
        for i, (chx_r, np_r) in enumerate(zip(chainerx_result, numpy_result)):
            _check_chainerx_numpy_result_impl(
                check_result_func, chx_r, np_r, indices + (i,))

    elif isinstance(chainerx_result, chainerx.ndarray):
        _check_chainerx_numpy_result_array(
            check_result_func, chainerx_result, numpy_result, indices)

    else:
        if _is_numpy_type(chainerx_result):
            raise _ResultsCheckFailure(
                'chainerx result should not be a NumPy type', indices)
        if type(chainerx_result) != type(numpy_result):
            raise _ResultsCheckFailure('Type mismatch', indices)
        if chainerx_result != numpy_result:
            raise _ResultsCheckFailure('Not equal', indices)


def _check_chainerx_numpy_result(
        check_result_func, chainerx_result, numpy_result):
    # Catch _ResultsCheckFailure and generate a comprehensible error message.
    try:
        _check_chainerx_numpy_result_impl(
            check_result_func, chainerx_result, numpy_result, indices=())
    except _ResultsCheckFailure as e:
        indices = e.indices
        chx_r = chainerx_result
        np_r = numpy_result
        i = 0
        while len(indices[i:]) > 0:
            chx_r = chx_r[indices[i]]
            np_r = np_r[indices[i]]
            i += 1

        def make_message(e):
            indices_str = ''.join('[{}]'.format(i) for i in indices)
            s = '{}: {}\n\n'.format(e.msg, e.condense_results(np_r, chx_r))
            if len(indices) > 0:
                s += 'chainerx results{}: {}\n'.format(
                    indices_str, type(chx_r))
                s += '{}\n\n'.format(chx_r)
                s += 'numpy results{}: {}\n'.format(indices_str, type(np_r))
                s += '{}\n\n'.format(np_r)
            s += 'chainerx results: {}\n'.format(type(chainerx_result))
            s += '{}\n\n'.format(chainerx_result)
            s += 'numpy results: {}\n'.format(type(numpy_result))
            s += '{}\n\n'.format(numpy_result)
            return s

        raise AssertionError(make_message(e))


def _make_decorator(check_result_func, name, accept_error):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(*args, **kw):
            kw[name] = chainerx
            chainerx_result, chainerx_error, chainerx_tb = _call_func(
                impl, args, kw)

            kw[name] = numpy
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                numpy_result, numpy_error, numpy_tb = _call_func(
                    impl, args, kw)

            if chainerx_error or numpy_error:
                _check_chainerx_numpy_error(chainerx_error, chainerx_tb,
                                            numpy_error, numpy_tb,
                                            accept_error=accept_error)
                return
            assert chainerx_result is not None and numpy_result is not None, (
                'Either or both of ChainerX and numpy returned None. '
                'chainerx: {}, numpy: {}'.format(
                    chainerx_result, numpy_result))
            _check_chainerx_numpy_result(
                check_result_func, chainerx_result, numpy_result)
        # Apply dummy parametrization on `name` (e.g. 'xp') to avoid pytest
        # error when collecting test functions.
        return pytest.mark.parametrize(name, [None])(test_func)
    return decorator


def numpy_chainerx_allclose(**kwargs):
    """numpy_chainerx_allclose(
           *, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True,
           name='xp', dtype_check=True, strides_check=True, accept_error=())

    Decorator that checks that NumPy and ChainerX results are equal up to a
    tolerance.

    Args:
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         equal_nan(bool): Allow NaN values if True. Otherwise, fail the
             assertion if any NaN is found.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either ``numpy`` or
             ``chainerx`` module.
         dtype_check(bool): If ``True``, consistency of dtype is also checked.
             Disabling ``dtype_check`` also implies ``strides_check=False``.
         strides_check(bool): If ``True``, consistency of strides is also
             checked.
         accept_error(Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and ChainerX test raises
             the same type of errors, and the type of the errors is specified
             with this argument, the errors are ignored and not raised.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`numpy_chainerx_allclose`
    (except the type of array module) even if ``xp`` is ``numpy`` or
    ``chainerx``.

    .. seealso:: :func:`chainerx.testing.assert_allclose_ex`
    """  # NOQA
    rtol = kwargs.pop('rtol', 1e-7)
    atol = kwargs.pop('atol', 0)
    equal_nan = kwargs.pop('equal_nan', True)
    err_msg = kwargs.pop('err_msg', '')
    verbose = kwargs.pop('verbose', True)
    name = kwargs.pop('name', 'xp')
    dtype_check = kwargs.pop('dtype_check', None)
    strides_check = kwargs.pop('strides_check', None)
    accept_error = kwargs.pop('accept_error', ())

    def check_result_func(x, y):
        array.assert_allclose_ex(
            x, y, rtol, atol, equal_nan, err_msg, verbose,
            dtype_check=dtype_check, strides_check=strides_check)

    return _make_decorator(check_result_func, name, accept_error)


def numpy_chainerx_array_equal(**kwargs):
    """numpy_chainerx_array_equal(
           *, err_msg='', verbose=True, name='xp', dtype_check=True,
           strides_check=True, accept_error=()):

    Decorator that checks that NumPy and ChainerX results are equal.

    Args:
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either ``numpy`` or
             ``chainerx`` module.
         dtype_check(bool): If ``True``, consistency of dtype is also checked.
             Disabling ``dtype_check`` also implies ``strides_check=False``
         strides_check(bool): If ``True``, consistency of strides is also
             checked.
         accept_error(Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and ChainerX test raises
             the same type of errors, and the type of the errors is specified
             with this argument, the errors are ignored and not raised.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`numpy_chainerx_array_equal`
    (except the type of array module) even if ``xp`` is ``numpy`` or
    ``chainerx``.

    .. seealso:: :func:`chainerx.testing.assert_array_equal_ex`
    """
    err_msg = kwargs.pop('err_msg', '')
    verbose = kwargs.pop('verbose', True)
    name = kwargs.pop('name', 'xp')
    dtype_check = kwargs.pop('dtype_check', None)
    strides_check = kwargs.pop('strides_check', None)
    accept_error = kwargs.pop('accept_error', ())

    def check_result_func(x, y):
        array.assert_array_equal_ex(
            x, y, err_msg, verbose, dtype_check=dtype_check,
            strides_check=strides_check)

    return _make_decorator(check_result_func, name, accept_error)
