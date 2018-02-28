import pytest

import xchainer


def test_name_native():
    backend = xchainer.get_global_default_context().get_backend('native')
    assert 'native' == backend.name


def test_get_device_native():
    backend = xchainer.get_global_default_context().get_backend('native')
    device = backend.get_device(0)
    assert 0 == device.index
    assert 'native:0' == device.name
    assert device is backend.get_device(0)


@pytest.mark.cuda
def test_name_cuda():
    backend = xchainer.get_global_default_context().get_backend('cuda')
    assert 'cuda' == backend.name


@pytest.mark.cuda
def test_get_device_cuda():
    backend = xchainer.get_global_default_context().get_backend('cuda')
    device = backend.get_device(0)
    assert 0 == device.index
    assert 'cuda:0' == device.name
    assert device is backend.get_device(0)
