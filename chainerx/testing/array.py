import numpy.testing

import chainerx

# NumPy-like assertion functions that accept both NumPy and ChainerX arrays


def _as_numpy(x):
    if isinstance(x, chainerx.ndarray):
        return chainerx.to_numpy(x)
    assert isinstance(x, numpy.ndarray) or numpy.isscalar(x)
    return x


def _check_dtype_and_strides(x, y, dtype_check, strides_check):
    if (strides_check is not None
            and dtype_check is not None
            and strides_check
            and not dtype_check):
        raise ValueError(
            'Combination of dtype_check=False and strides_check=True is not '
            'allowed')
    if dtype_check is None:
        dtype_check = True
    if strides_check is None:
        strides_check = dtype_check

    if (isinstance(x, (numpy.ndarray, chainerx.ndarray))
            and isinstance(y, (numpy.ndarray, chainerx.ndarray))):
        if strides_check:
            assert x.strides == y.strides, (
                'Strides mismatch: x: {}, y: {}'.format(x.strides, y.strides))
        if dtype_check:
            assert x.dtype.name == y.dtype.name, (
                'Dtype mismatch: x: {}, y: {}'.format(x.dtype, y.dtype))


def _preprocess_input(a):
    # Convert chainerx.Scalar to Python scalar
    if isinstance(a, chainerx.Scalar):
        a = a.tolist()

    # Convert to something NumPy can handle and return
    return _as_numpy(a)


def assert_allclose(
        x, y, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True):
    """Raises an AssertionError if two array_like objects are not equal up to a
    tolerance.

    Args:
         x(numpy.ndarray or chainerx.ndarray): The actual object to check.
         y(numpy.ndarray or chainerx.ndarray): The desired, expected object.
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         equal_nan(bool): Allow NaN values if True. Otherwise, fail the
             assertion if any NaN is found.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
    .. seealso:: :func:`numpy.testing.assert_allclose`
    """
    x = _preprocess_input(x)
    y = _preprocess_input(y)

    numpy.testing.assert_allclose(
        x, y, rtol=rtol, atol=atol, equal_nan=equal_nan, err_msg=err_msg,
        verbose=verbose)


def assert_array_equal(x, y, err_msg='', verbose=True):
    """Raises an AssertionError if two array_like objects are not equal.

    Args:
         x(numpy.ndarray or chainerx.ndarray): The actual object to check.
         y(numpy.ndarray or chainerx.ndarray): The desired, expected object.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
    .. seealso:: :func:`numpy.testing.assert_array_equal`
    """
    x = _preprocess_input(x)
    y = _preprocess_input(y)

    numpy.testing.assert_array_equal(x, y, err_msg=err_msg, verbose=verbose)


def assert_allclose_ex(x, y, *args, **kwargs):
    """assert_allclose_ex(
           x, y, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True,
           *, dtype_check=True, strides_check=True)

    Raises an AssertionError if two array_like objects are not equal up to a
    tolerance.

    Args:
         x(numpy.ndarray or chainerx.ndarray): The actual object to check.
         y(numpy.ndarray or chainerx.ndarray): The desired, expected object.
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         equal_nan(bool): Allow NaN values if True. Otherwise, fail the
             assertion if any NaN is found.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
         dtype_check(bool): If ``True``, consistency of dtype is also checked.
             Disabling ``dtype_check`` also implies ``strides_check=False``.
         strides_check(bool): If ``True``, consistency of strides is also
             checked.
    .. seealso:: :func:`numpy.testing.assert_allclose`
    """
    dtype_check = kwargs.pop('dtype_check', None)
    strides_check = kwargs.pop('strides_check', None)

    assert_allclose(x, y, *args, **kwargs)
    _check_dtype_and_strides(x, y, dtype_check, strides_check)


def assert_array_equal_ex(x, y, *args, **kwargs):
    """assert_array_equal_ex(
           x, y, err_msg='', verbose=True, *, dtype_check=True,
           strides_check=True)

    Raises an AssertionError if two array_like objects are not equal.

    Args:
         x(numpy.ndarray or chainerx.ndarray): The actual object to check.
         y(numpy.ndarray or chainerx.ndarray): The desired, expected object.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
         dtype_check(bool): If ``True``, consistency of dtype is also checked.
             Disabling ``dtype_check`` also implies ``strides_check=False``.
         strides_check(bool): If ``True``, consistency of strides is also
             checked.
    .. seealso::
       :func:`numpy.testing.assert_array_equal`
    """
    dtype_check = kwargs.pop('dtype_check', None)
    strides_check = kwargs.pop('strides_check', None)

    assert_array_equal(x, y, *args, **kwargs)
    _check_dtype_and_strides(x, y, dtype_check, strides_check)
