import pytest

import xchainer


CPU = xchainer.Device('cpu')
CUDA = xchainer.Device('cuda')


def test_device():
    cpu1 = xchainer.Device('cpu')
    cpu2 = xchainer.Device('cpu')
    cuda = xchainer.Device('cuda')
    assert cpu1 == cpu2
    assert not (cpu1 != cpu2)
    assert not (cpu1 == cuda)
    assert cpu1 != cuda

    with pytest.raises(xchainer.DeviceError):
        xchainer.Device('a' * 8)  # too long device name


def test_current_device():
    device = xchainer.get_current_device()

    xchainer.set_current_device('cpu')
    assert xchainer.get_current_device() == CPU

    xchainer.set_current_device('cuda')
    assert xchainer.get_current_device() == CUDA

    with pytest.raises(xchainer.DeviceError):
        xchainer.set_current_device('invalid_device')

    xchainer.set_current_device(device)


def test_device_scope():
    device = xchainer.get_current_device()

    xchainer.set_current_device('cpu')
    with xchainer.device_scope('cuda'):
        assert xchainer.get_current_device() == CUDA

    scope = xchainer.device_scope('cuda')
    assert xchainer.get_current_device() == CPU
    with scope:
        assert xchainer.get_current_device() == CUDA
    assert xchainer.get_current_device() == CPU

    xchainer.set_current_device(device)
