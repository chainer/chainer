import numpy.testing

import xchainer

# NumPy-like assertion functions that accept both NumPy and xChainer arrays


def _check_xchainer_array(x):
    # Checks basic conditions that are assumed to hold true for any given xChainer array passed to assert_array_close and
    # assert_array_equal.
    assert isinstance(x, xchainer.ndarray)
    assert not x.is_grad_required()


def _as_numpy(x):
    if isinstance(x, xchainer.ndarray):
        # TODO(hvy): Use a function that converts xChainer arrays to NumPy arrays.
        return x.to_device('native:0')
    assert isinstance(x, numpy.ndarray) or numpy.isscalar(x)
    return x


def _preprocess_input(a):
    # Convert xchainer.Scalar to Python scalar
    if isinstance(a, xchainer.Scalar):
        a = a.tolist()

    # Check conditions for xchainer.ndarray
    if isinstance(a, xchainer.ndarray):
        _check_xchainer_array(a)

    # Convert to something NumPy can handle
    a = _as_numpy(a)
    return a


def assert_allclose(x, y, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True):
    """Raises an AssertionError if two array_like objects are not equal up to a tolerance.

    Args:
         x(numpy.ndarray or xchainer.ndarray): The actual object to check.
         y(numpy.ndarray or xchainer.ndarray): The desired, expected object.
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         equal_nan(bool): Allow NaN values if True. Otherwise, fail the assertion if any NaN is found.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
    .. seealso:: :func:`numpy.testing.assert_allclose`
    """
    x = _preprocess_input(x)
    y = _preprocess_input(y)

    # TODO(sonots): Uncomment after strides compatibility between xChainer and NumPy is implemented.
    # assert x.strides == y.strides

    numpy.testing.assert_allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan, err_msg=err_msg, verbose=verbose)


def assert_array_equal(x, y, err_msg='', verbose=True):
    """Raises an AssertionError if two array_like objects are not equal.

    Args:
         x(numpy.ndarray or xchainer.ndarray): The actual object to check.
         y(numpy.ndarray or xchainer.ndarray): The desired, expected object.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
    .. seealso:: :func:`numpy.testing.assert_array_equal`
    """
    x = _preprocess_input(x)
    y = _preprocess_input(y)

    # TODO(sonots): Uncomment after strides compatibility between xChainer and NumPy is implemented.
    # assert x.strides == y.strides

    numpy.testing.assert_array_equal(x, y, err_msg=err_msg, verbose=verbose)
