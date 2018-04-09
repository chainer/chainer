import functools
import traceback
import warnings

import numpy
import pytest

import xchainer
from xchainer.testing import array

_float_dtypes = [
    xchainer.float32,
    xchainer.float64,
]


_signed_dtypes = [
    xchainer.int8,
    xchainer.int16,
    xchainer.int32,
    xchainer.int64,
    xchainer.float32,
    xchainer.float64,
]


_unsigned_dtypes = [
    xchainer.uint8,
]


def _call_func(impl, args, kw):
    try:
        result = impl(*args, **kw)
        assert result is not None
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


def _make_positive_indices(impl, args, kw):
    ks = [k for k, v in kw.items() if v in _unsigned_dtypes]
    for k in ks:
        kw[k] = numpy.intp
    mask = numpy.array(impl(*args, **kw)) >= 0
    return numpy.nonzero(mask)


def _contains_signed_and_unsigned(kw):
    vs = list(kw.values())
    return any(d in vs for d in _unsigned_dtypes) and \
        any(d in vs for d in _float_dtypes + _signed_dtypes)


def _make_decorator(check_func, name, device_arg, device_check, type_check, accept_error):
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

            assert xchainer_result.shape == numpy_result.shape

            # Behavior of assigning a negative value to an unsigned integer
            # variable is undefined.
            # nVidia GPUs and Intel CPUs behave differently.
            # To avoid this difference, we need to ignore dimensions whose
            # values are negative.
            skip = False
            if _contains_signed_and_unsigned(kw) and \
                    xchainer_result.dtype in _unsigned_dtypes:
                inds = _make_positive_indices(impl, args, kw)
                if xchainer_result.shape == ():
                    skip = inds[0].size == 0
                else:
                    xchainer_result = numpy.array(xchainer_result)[inds]
                    numpy_result = numpy.array(numpy_result)[inds]

            if not skip:
                check_func(xchainer_result, numpy_result)
            if type_check:
                assert numpy.dtype(xchainer_result.dtype.name) == numpy_result.dtype
            if device_check and device_arg in kw:
                assert xchainer_result.device is kw[device_arg]
        return test_func
    return decorator


def numpy_xchainer_array_equal(*, err_msg='', verbose=True, name='xp', device_arg='device',
                               rtol=0, atol=0, device_check=True, type_check=True, accept_error=()):
    """Decorator that checks NumPy results and xChainer ones are equal.

    Args:
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either ``numpy`` or ``xchainer`` module.
         device_arg(str): Argument name whose value is device
         device_check(bool): If ``True``, check equality between device of xchainer array
             and the device argument.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and xChainer test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`numpy_xchainer_array_equal`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``xchainer``.

    .. seealso:: :func:`xchainer.testing.assert_array_equal`
    """
    def check_func(x, y):
        array.assert_array_equal(x, y, rtol, atol, err_msg, verbose)

    return _make_decorator(check_func, name, device_arg, device_check, type_check, accept_error)
