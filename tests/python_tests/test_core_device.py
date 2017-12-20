import xchainer

import pytest


def test_device():
    device = xchainer.get_current_device()

    xchainer.set_current_device('cpu')
    assert str(xchainer.get_current_device()) == '<Device cpu>'

    xchainer.set_current_device('cuda')
    assert str(xchainer.get_current_device()) == '<Device cuda>'

    with pytest.raises(xchainer.DeviceError):
        xchainer.set_current_device('invalid_device')

    xchainer.set_current_device(device)
