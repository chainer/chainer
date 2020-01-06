import numpy as np


def shaped_range(*shape, dtype=np.float32):
    r = np.arange(np.prod(shape))
    r = r.reshape(shape)
    return r


def _increasing_impl(*shape, dtype=np.float32, negative=True, bias=0):
    r = shaped_range(*shape, dtype=dtype)
    if negative:
        r -= r.size // 2
    if dtype in (np.float32, np.float64):
        r = r * 0.5
    r += bias
    return r.astype(dtype)


def increasing(*shape, dtype=np.float32):
    """Returns a monotonically increasing ndarray for test inputs.

    The output will contain both zero, negative numbers, and non
    integer numbers for float dtypes. A test writer is supposed to
    consider this function first.

    Example:

    >>> onnx_chainer.testing.input_generator.increasing(3, 4)
    array([[-3. , -2.5, -2. , -1.5],
           [-1. , -0.5,  0. ,  0.5],
           [ 1. ,  1.5,  2. ,  2.5]], dtype=float32)

    Args:
        shape (tuple of int): The shape of the output array.
        dtype (numpy.dtype): The dtype of the output array.

    Returns:
        numpy.ndarray
    """
    return _increasing_impl(*shape, dtype=dtype)


def nonzero_increasing(*shape, dtype=np.float32, bias=1e-7):
    """Returns a monotonically increasing ndarray for test inputs.

    Similar to `increasing` but contains no zeros. Expected to be used
    for divisors.

    Example:

    >>> onnx_chainer.testing.input_generator.nonzero_increasing(3, 4)
    array([[-3.0000000e+00, -2.5000000e+00, -1.9999999e+00, -1.4999999e+00],
           [-9.9999988e-01, -4.9999991e-01,  1.0000000e-07,  5.0000012e-01],
           [ 1.0000001e+00,  1.5000001e+00,  2.0000000e+00,  2.5000000e+00]],
          dtype=float32)

    Args:
        shape (tuple of int): The shape of the output array.
        dtype (numpy.dtype): The dtype of the output array.
        bias (float): The bias to avoid zero.

    Returns:
        numpy.ndarray
    """
    assert dtype in (np.float32, np.float64)
    return _increasing_impl(*shape, dtype=dtype, bias=bias)


def positive_increasing(*shape, dtype=np.float32, bias=1e-7):
    """Returns a monotonically increasing ndarray for test inputs.

    Similar to `increasing` but contains only positive numbers. Expected
    to be used for `math.log`, `math.sqrt`, etc.

    Example:

    >>> onnx_chainer.testing.input_generator.positive_increasing(3, 4)
    array([[1.0000000e-07, 5.0000012e-01, 1.0000001e+00, 1.5000001e+00],
           [2.0000000e+00, 2.5000000e+00, 3.0000000e+00, 3.5000000e+00],
           [4.0000000e+00, 4.5000000e+00, 5.0000000e+00, 5.5000000e+00]],
          dtype=float32)

    Args:
        shape (tuple of int): The shape of the output array.
        dtype (numpy.dtype): The dtype of the output array.
        bias (float): The bias to avoid zero.

    Returns:
        numpy.ndarray
    """
    return _increasing_impl(*shape, dtype=dtype, negative=False, bias=bias)
