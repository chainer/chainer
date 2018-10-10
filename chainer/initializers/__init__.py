import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.initializers.constant import Constant
from chainer.initializers.constant import Identity  # NOQA
from chainer.initializers.constant import NaN  # NOQA
from chainer.initializers.constant import One  # NOQA
from chainer.initializers.constant import Zero  # NOQA
from chainer.initializers.normal import GlorotNormal  # NOQA
from chainer.initializers.normal import HeNormal  # NOQA
from chainer.initializers.normal import LeCunNormal
from chainer.initializers.normal import Normal  # NOQA
from chainer.initializers.orthogonal import Orthogonal  # NOQA
from chainer.initializers.uniform import GlorotUniform  # NOQA
from chainer.initializers.uniform import HeUniform  # NOQA
from chainer.initializers.uniform import LeCunUniform  # NOQA
from chainer.initializers.uniform import Uniform  # NOQA
import chainerx


def generate_array(initializer, shape, xp, dtype=None, device=None):
    """Return initialized array.

    The algorithms used to make the new values depend on the
    concrete derived classes. If the initializer has the ``dtype`` attribute,
    it is used to construct the array. Otherwise, ``chainer.config.dtype`` is
    used instead. See :ref:`configuration` for the dtype config.

    Args:
        initializer: A callable object that takes :class:`numpy.ndarray`
             or :class:`cupy.ndarray` and edits its value.
        shape (tuple): Shape of a return array.
        xp (module): :mod:`cupy`, :mod:`numpy`, or :mod:`chainerx`.
        dtype: Dtype specifier. If omitted, ``initializer.dtype`` is used.
        device: Target device specifier. If omitted, the current device is
             used for :mod:`cupy`, and the default device is used for
             :mod:`chainerx`.

    Returns:
        numpy.ndarray or cupy.ndarray: An initialized array.

    """
    dtype_attr = getattr(initializer, 'dtype', None)
    if dtype is not None and dtype_attr is not None \
            and numpy.dtype(dtype) != numpy.dtype(dtype_attr):
        raise ValueError(
            'dtype mismatch: {} != {}'.format(dtype, dtype_attr))
    if dtype is None:
        dtype = dtype_attr
    dtype = chainer.get_dtype(dtype)

    if device is None:
        if xp is cuda.cupy:
            device = cuda.Device().id
        elif xp is chainerx:
            device = chainerx.get_default_device()

    if xp is chainerx:
        # TODO(sonots): Directly use initializer after ChainerX
        # supports random.
        device = chainerx.get_device(device)
        array = xp.empty(shape, dtype=dtype, device=device)
        if device.backend.name == 'native':
            array = backend.to_numpy(array)
        elif device.backend.name == 'cuda':
            array = cuda.to_gpu(array)
        else:
            raise RuntimeError('ChainerX backend: {} is not supported.'.format(
                device.backend.name))
        initializer(array)
        return backend.to_chainerx(array, device=device)

    with cuda.get_device_from_id(device):
        array = xp.empty(shape, dtype=dtype)
    initializer(array)
    return array


def _get_initializer(initializer):
    if initializer is None:
        return LeCunNormal()
    if numpy.isscalar(initializer):
        return Constant(initializer)
    if isinstance(initializer, numpy.ndarray):
        return Constant(initializer)

    if not callable(initializer):
        raise TypeError('invalid type of initializer: %s' % type(initializer))
    return initializer
