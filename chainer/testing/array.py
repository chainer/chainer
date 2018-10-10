import numpy
import six

from chainer import backend
from chainer import utils
import chainerx


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
    x = backend.to_numpy(utils.force_array(x))
    y = backend.to_numpy(utils.force_array(y))
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


def _as_noncontiguous_array(array):
    def as_noncontiguous_array(a):
        if a is None:
            return None

        xp = backend.get_array_module(a)
        if xp is not chainerx:
            ret = xp.empty(
                (a.shape[0] * 2,) + a.shape[1:], dtype=a.dtype)
            ret[::2] = a
            ret = ret[::2]
            assert not ret.flags.c_contiguous
        else:
            # chainerx.ndarray does not support item assignment.
            if a.ndim == 1:
                ret = xp.diag(
                    xp.diag(a, device=a.device), device=a.device)
            else:
                ret = a.T.copy().T
            assert not ret.is_contiguous

        return ret
    return backend._obj_to_array(array, as_noncontiguous_array)
