import numpy.testing

import xchainer

# NumPy-like assertion functions that accept both NumPy and xChainer arrays


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
    numpy.testing.assert_array_equal(
        numpy.array(x), numpy.array(y), err_msg=err_msg,
        verbose=verbose)
