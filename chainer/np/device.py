import numpy

import chainer
from chainer.backends import cuda


class Device(object):

    """Object that represents a device.

    A device object is specified by the backend name and the device index.
    The backend name is either ``'native'`` or ``'cuda'``.
    The index is used to indentify the individual device handled by the
    backend.

    The device is identified by the pair of the backend name and the device
    index, or the concatenation of them separated by ``':'``. For example,
    the second device of CUDA backend is specified by ``('cuda', 1)`` or
    ``'cuda:1'`` (note that the device index is zero-originated).

    Note: it is not recommended to directly create an instance of this class.
    Use :func:`~chainer.np.get_device` instead.

    Attributes:
        backend (str): Name of the backend.
        index (int): Index of the device.
        xp: NumPy-compatible module for the device.
        underlying_device: Backend-specific underlying device object. It
            supports ``use()`` method and the context management protocol
            (``__enter__`` and ``__exit__``) to temporarily make the device
            current for the backend-specific APIs.

    """

    def __init__(self, backend, index, xp, underlying_device):
        self.backend = backend
        self.index = index
        self.xp = xp
        self.underlying_device = underlying_device
        self._device_stack = []

    def __str__(self):
        return '%s:%d' % (self.backend, self.index)

    def __repr__(self):
        return 'Device("%s", %d)' % (self.backend, self.index)

    def __enter__(self):
        self._device_stack.append(chainer.config.default_device)
        chainer.config.default_device = self

        self.underlying_device.__enter__()
        return self

    def __exit__(self, *exc_info):
        self.underlying_device.__exit__(*exc_info)
        chainer.config.default_device = self._device_stack.pop()

    def use(self):
        chainer.config.default_device = self
        self.underlying_device.use()


_devices = {}


def get_device(name=None, index=None):
    """Returns the device object specified by the name and index.

    Args:
        name (str or chainer.np.Device or None): Device or backend name. If it
            is ``None``, the default device is returned. If it is already a
            ``Device`` object, this argument is returned as is.
        index (int or None): Index of the device. If ``name`` also contains the
            index in its name (specified in the form
            ``'<backend name>:<index>'``), this argument can be omitted.
            If ``name`` only represents the backend name and still this
            argument is omitted, ``0`` is used.

    Returns:
        chainer.np.Device: Device object for the specified backend name and
            index.

    """
    if name is None:
        return chainer.config.default_device
    if isinstance(name, Device):
        return name

    pair = _resolve_device_name(name, index)
    if pair in _devices:
        return _devices[pair]

    backend, index = pair
    if backend == 'native':
        device = Device('native', index, numpy, cuda.DummyDeviceType())
    elif backend == 'cuda':
        device = Device('cuda', index, cuda.cupy, cuda.Device(index))
    else:
        raise ValueError('invalid backend name: "{}"'.format(backend))

    _devices[pair] = device
    return device


def get_default_device():
    """Returns the default device of the current thread."""
    return chainer.config.default_device


def set_default_device(name, index=None):
    """Sets the default device for the current thread.

    Args:
        name (str or chainer.np.Device): The device to set. It can be either
            a backend name, device name, or a :class:`Device` object.
            If it is a backend name, ``index`` is used for the device index.
        index (int or None): The device index. If it is omitted and ``name``
            does not specify the index, ``0`` is used.

    """
    get_device(name, index).use()


def _resolve_device_name(name, index=None):
    parts = name.split(':')
    if len(parts) == 0:
        raise ValueError('device name is empty')

    if len(parts) == 1:
        if index is None:
            index = 0
        return parts[0], index

    try:
        n, i = parts
    except ValueError:
        raise ValueError('invalid device name: "{}"'.format(name))

    if index is not None:
        raise ValueError(
            'device id is duplicated: backend={}, index={}'.format(
                name, index))
    return n, int(i)
