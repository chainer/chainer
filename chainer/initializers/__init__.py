import typing as tp  # NOQA

import numpy

import chainer
from chainer import backend
from chainer.backends import _chainerx  # NOQA

# import class and function
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
from chainer.initializers.sampling import DownsamplingConvFilter  # NOQA
from chainer.initializers.sampling import UpsamplingDeconvFilter  # NOQA
from chainer.initializers.uniform import GlorotUniform  # NOQA
from chainer.initializers.uniform import HeUniform  # NOQA
from chainer.initializers.uniform import LeCunUniform  # NOQA
from chainer.initializers.uniform import Uniform  # NOQA
from chainer import types  # NOQA


def generate_array(
        initializer: types.AbstractInitializer,
        shape: types.ShapeSpec,
        xp: types.Xp,
        dtype: tp.Optional[types.DTypeSpec] = None,
        device: tp.Optional[types.DeviceSpec] = None
) -> types.NdArray:
    """Return initialized array.

    The algorithms used to make the new values depend on the
    concrete derived classes. If the initializer has the ``dtype`` attribute,
    it is used to construct the array. Otherwise, ``chainer.config.dtype`` is
    used instead. See :ref:`configuration` for the dtype config.

    Args:
        initializer: A callable object that takes :ref:`ndarray` and edits its
            value.
        shape (int or tuple of int): Shape of the initialized array.
        xp (module): :mod:`cupy`, :mod:`numpy`, or :mod:`chainerx`.
        dtype: Dtype specifier. If omitted, ``initializer.dtype`` is used.
        device: Target device specifier. If omitted, the current device is
             used for :mod:`cupy`, and the default device is used for
             :mod:`chainerx`.

    Returns:
        :ref:`ndarray`: An initialized array.

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
        backend_device = backend._guess_device_from_array_module(xp)
    else:
        backend_device = chainer.get_device(device)
        if xp != backend_device.xp:
            raise ValueError('xp and device arguments are inconsistent.')
    with chainer.using_device(backend_device):
        array = xp.empty(shape, dtype=dtype)
        initializer(array)
    return array


def _get_initializer(
        initializer: tp.Optional[types.InitializerSpec]
) -> types.AbstractInitializer:
    if initializer is None:
        return LeCunNormal()
    if (isinstance(initializer, chainer.get_array_types())
            or numpy.isscalar(initializer)):
        return Constant(initializer)

    if not callable(initializer):
        raise TypeError('invalid type of initializer: %s' % type(initializer))
    return initializer


def _check_is_initializer_like(initializer):
    if not (initializer is None
            or isinstance(initializer, chainer.Initializer)
            or callable(initializer)
            or isinstance(initializer, chainer.get_array_types())
            or numpy.isscalar(initializer)):
        raise TypeError(
            'Initializer is of wrong type: {}. Allowed types are Initializer, '
            'ndarray and scalar.'.format(type(initializer)))
