import numpy.testing

import xchainer

# NumPy-like assertion functions that accept both NumPy and xChainer arrays


def assert_allclose(x, y, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True):
    """Raises an AssertionError if two array_like objects are not equal up to a tolerance.

    Args:
         x(numpy.ndarray or xchainer.Array): The actual object to check.
         y(numpy.ndarray or xchainer.Array): The desired, expected object.
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         equal_nan(bool): Allow NaN values if True. Otherwise, fail the assertion if any NaN is found.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
    .. seealso:: :func:`numpy.testing.assert_allclose`
    """
    # TODO(hvy): Uncomment after strides compatibility between xchaienr and numpy is implemented.
    # assert x.strides == y.strides
    # TODO(hvy): Remove following explicit `to_device` transfer if conversion from
    # xchainer.Array to numpy.ndarray via buffer protocol supports the device transfer.
    if isinstance(x, xchainer.Array):
        assert not x.is_grad_required()
        x = x.to_device('native:0')
    if isinstance(y, xchainer.Array):
        assert not x.is_grad_required()
        y = y.to_device('native:0')
    numpy.testing.assert_allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan, err_msg=err_msg, verbose=verbose)


def assert_array_equal(x, y, err_msg='', verbose=True):
    """Raises an AssertionError if two array_like objects are not equal.

    Args:
         x(numpy.ndarray or xchainer.Array): The actual object to check.
         y(numpy.ndarray or xchainer.Array): The desired, expected object.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
    .. seealso:: :func:`numpy.testing.assert_array_equal`
    """
    # TODO(sonots): Uncomment after strides compatibility between xchaienr and numpy is implemented.
    # assert x.strides == y.strides
    # TODO(sonots): Remove following explicit `to_device` transfer if conversion from
    # xchainer.Array to numpy.ndarray via buffer protocol supports the device transfer.
    if isinstance(x, xchainer.Array):
        assert not x.is_grad_required()
        x = x.to_device('native:0')
    if isinstance(y, xchainer.Array):
        assert not x.is_grad_required()
        y = y.to_device('native:0')
    numpy.testing.assert_array_equal(x, y, err_msg=err_msg, verbose=verbose)
