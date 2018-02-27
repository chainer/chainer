import xchainer


def test_name():
    backend = xchainer.get_global_default_context().get_backend('native')
    assert 'native' == backend.name


def test_get_device():
    backend = xchainer.get_global_default_context().get_backend('native')
    device = backend.get_device(0)
    assert 0 == device.index
    assert 'native:0' == device.name
    assert device is backend.get_device(0)
