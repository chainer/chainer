import pytest

import chainerx


def test_name_native():
    backend = chainerx.get_global_default_context().get_backend('native')
    assert 'native' == backend.name


def test_get_device_native():
    backend = chainerx.get_global_default_context().get_backend('native')
    device = backend.get_device(0)
    assert 0 == device.index
    assert 'native:0' == device.name
    assert device is backend.get_device(0)


def test_get_device_count_native():
    backend = chainerx.get_global_default_context().get_backend('native')
    assert backend.get_device_count() > 0


@pytest.mark.cuda
def test_name_cuda():
    backend = chainerx.get_global_default_context().get_backend('cuda')
    assert 'cuda' == backend.name


@pytest.mark.cuda
def test_get_device_cuda():
    backend = chainerx.get_global_default_context().get_backend('cuda')
    device = backend.get_device(0)
    assert 0 == device.index
    assert 'cuda:0' == device.name
    assert device is backend.get_device(0)


@pytest.mark.cuda
def test_get_device_count_cuda():
    backend = chainerx.get_global_default_context().get_backend('cuda')
    assert backend.get_device_count() > 0
