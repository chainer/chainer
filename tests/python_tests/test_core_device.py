import pytest

import xchainer


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
    assert str(xchainer.get_current_device()) == '<Device cpu>'

    xchainer.set_current_device('cuda')
    assert str(xchainer.get_current_device()) == '<Device cuda>'

    with pytest.raises(xchainer.DeviceError):
        xchainer.set_current_device('invalid_device')

    xchainer.set_current_device(device)
