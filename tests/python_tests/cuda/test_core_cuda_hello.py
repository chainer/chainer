import pytest

import xchainer


cuda = pytest.mark.cuda


@cuda
def test_hello(capfd):
    device = xchainer.get_current_device()

    xchainer.set_current_device('cpu')
    xchainer.hello()
    out, _ = capfd.readouterr()
    assert out == 'Hello, World!\n'

    xchainer.set_current_device('cuda')
    xchainer.hello()
    out, _ = capfd.readouterr()
    assert out == 'Hello, CUDA!\n'

    xchainer.set_current_device(device)
