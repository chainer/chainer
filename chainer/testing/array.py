import numpy
import six

import chainer
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
    x = backend.CpuDevice().send(utils.force_array(x))
    y = backend.CpuDevice().send(utils.force_array(y))
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
            xx = numpy.atleast_1d(x)
            yy = numpy.atleast_1d(y)
            err = numpy.abs(xx - yy)
            tol_err = atol + rtol * numpy.abs(yy).astype(numpy.float64)
            i = numpy.unravel_index(
                numpy.argmax(err.astype(numpy.float64) - tol_err), err.shape)
            if yy[i] == 0:
                rel_err = 'inf'
            else:
                rel_err = err[i] / numpy.abs(yy[i])
            f.write(
                '  i: {}\n'.format(i) +
                '  x[i]: {}\n'.format(xx[i]) +
                '  y[i]: {}\n'.format(yy[i]) +
                '  relative error[i]: {}\n'.format(rel_err) +
                '  absolute error[i]: {}\n'.format(err[i]))
        opts = numpy.get_printoptions()
        try:
            numpy.set_printoptions(threshold=10000)
            f.write('x: ' + numpy.array2string(x, prefix='x: ') + '\n')
            f.write('y: ' + numpy.array2string(y, prefix='y: ') + '\n')
        finally:
            numpy.set_printoptions(**opts)
        raise AssertionError(f.getvalue())


def _as_noncontiguous_array(array):
    # This is a temporary function used by tests to convert contiguous arrays
    # to non-contiguous arrays.
    #
    # This functions can be removed if e.g. BackendConfig starts supporting
    # contiguousness configurations and the array conversion method takes that
    # into account. Note that that would also mean rewriting tests to use the
    # backend injector in the first place.

    def as_noncontiguous_array(a):
        if a is None:
            return None

        if a.size <= 1:
            return a

        device = backend.get_device_from_array(a)
        xp = device.xp
        slices = (slice(None, None, 2),) * a.ndim
        with chainer.using_device(device):
            ret = xp.empty(tuple([s * 2 for s in a.shape]), dtype=a.dtype)
            ret[slices] = a
            ret = ret[slices]
        if device.xp is chainerx:
            assert not ret.is_contiguous
        else:
            assert not ret.flags.c_contiguous

        return ret

    if isinstance(array, (list, tuple)):
        return type(array)([_as_noncontiguous_array(arr) for arr in array])
    else:
        return as_noncontiguous_array(array)
