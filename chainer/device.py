import six

from chainer import cuda


def is_cpu(device):
    """Determines if the specified device is a CPU device.

    .. note::
        This function does not check whether the specified device is actually
        available.

    Args:
        device (device specifier): Device specifier.

    Returns:
        ``True`` if ``device`` is a CPU device. ``False`` otherwise.
    """
    if isinstance(device, six.integer_types):
        return device == -1
    elif isinstance(device, cuda.DummyDeviceType):
        return True
    elif isinstance(device, cuda.Device):
        return False
    else:
        raise TypeError('Invalid device specifier type: {}'.format(type(device)))


def is_cuda(device):
    """Determines if the specified device is a CUDA device.

    .. note::
        This function does not check whether the specified device is actually
        available.

    Args:
        device (device specifier): Device specifier.

    Returns:
        ``True`` if ``device`` is a CUDA device. ``False`` otherwise.
    """
    return not is_cpu(device)
