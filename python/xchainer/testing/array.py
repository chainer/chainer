import numpy.testing

import xchainer

# NumPy-like assertion functions that accept both NumPy and xChainer arrays


def assert_array_equal(x, y, rtol=1e-7, atol=0, err_msg='', verbose=True):
    """Raises an AssertionError if two array_like objects are not equal.

    Args:
         x(numpy.ndarray or xchainer.Array): The actual object to check.
         y(numpy.ndarray or xchainer.Array): The desired, expected object.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
    .. seealso:: :func:`numpy.testing.assert_allclose`
    """
    # TODO(sonots): Remove following explicit `to_device` transfer if conversion from
    # xchainer.Array to numpy.ndarray via buffer protocol supports the device transfer.
    if isinstance(x, xchainer.Array):
        x = x.to_device('native:0')
    if isinstance(y, xchainer.Array):
        y = y.to_device('native:0')
    numpy.testing.assert_allclose(
        x, y, rtol, atol, err_msg=err_msg, verbose=verbose)
