import numpy
import six

from chainer.backends import cuda
from chainer import utils


def assert_allclose(x, y, atol=None, rtol=None, verbose=True):
    """Asserts if some corresponding element of x and y differs too much.

    This function can handle both CPU and GPU arrays simultaneously.

    Args:
        x: Left-hand-side array.
        y: Right-hand-side array.
        atol (float): Absolute tolerance. If ``None``, the default value
            depending on the dtype is used.
        rtol (float): Relative tolerance. If ``None``, the default value
            depending on the dtype is used.
        verbose (bool): If ``True``, it outputs verbose messages on error.

    """
    x = cuda.to_cpu(utils.force_array(x))
    y = cuda.to_cpu(utils.force_array(y))

    if atol is None:
        if x.dtype == numpy.float16:
            atol = 1e-3
        else:
            atol = 1e-5
    if rtol is None:
        if x.dtype == numpy.float16:
            rtol = 1e-2
        else:
            rtol = 1e-4

    try:
        numpy.testing.assert_allclose(
            x, y, atol=atol, rtol=rtol, verbose=verbose)
    except AssertionError as e:
        f = six.StringIO()
        f.write(str(e) + '\n\n')
        f.write(
            'assert_allclose failed: \n' +
            '  shape: {} {}\n'.format(x.shape, y.shape) +
            '  dtype: {} {}\n'.format(x.dtype, y.dtype))
        if x.shape == y.shape:
            xx = x if x.ndim != 0 else x.reshape((1,))
            yy = y if y.ndim != 0 else y.reshape((1,))
            err = numpy.abs(xx - yy)
            i = numpy.unravel_index(numpy.argmax(err), err.shape)
            f.write(
                '  i: {}\n'.format(i) +
                '  x[i]: {}\n'.format(xx[i]) +
                '  y[i]: {}\n'.format(yy[i]) +
                '  err[i]: {}\n'.format(err[i]))
        opts = numpy.get_printoptions()
        try:
            numpy.set_printoptions(threshold=10000)
            f.write('x: ' + numpy.array2string(x, prefix='x: ') + '\n')
            f.write('y: ' + numpy.array2string(y, prefix='y: ') + '\n')
        finally:
            numpy.set_printoptions(**opts)
        raise AssertionError(f.getvalue())
