import numpy
import sys

from chainer import cuda
from chainer import utils


def assert_allclose(x, y, atol=1e-5, rtol=1e-4, verbose=True):
    """Asserts if some corresponding element of x and y differs too much.

    This function can handle both CPU and GPU arrays simultaneously.

    Args:
        x: Left-hand-side array.
        y: Right-hand-side array.
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.
        verbose (bool): If ``True``, it outputs verbose messages on error.

    """
    x = cuda.to_cpu(utils.force_array(x))
    y = cuda.to_cpu(utils.force_array(y))
    try:
        numpy.testing.assert_allclose(
            x, y, atol=atol, rtol=rtol, verbose=verbose)
    except AssertionError:
        err = numpy.abs(x - y)
        i = numpy.unravel_index(numpy.argmax(err), err.shape)
        f = sys.stderr
        f.write(
            'assert_allclose failed: \n' +
            '  shape: {}\n'.format(x.shape) +
            '  dtype: {} {}\n'.format(x.dtype, y.dtype) +
            '  i: {}\n'.format(i) +
            '  x[i]: {}\n'.format(x[i]) +
            '  y[i]: {}\n'.format(y[i]) +
            '  err[i]: {}\n'.format(err[i]))
        f.flush()
        opts = numpy.get_printoptions()
        try:
            numpy.set_printoptions(threshold=10000)
            f.write('x: ' + numpy.array2string(x, prefix='x: ') + '\n')
            f.write('y: ' + numpy.array2string(y, prefix='y: ') + '\n')
            f.flush()
        finally:
            numpy.set_printoptions(**opts)
        raise
