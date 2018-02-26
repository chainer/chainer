import xchainer


def test_name():
    assert 'native' == xchainer.get_global_default_context().get_backend('native').name


def test_get_device():
    device = xchainer.get_global_default_context().get_backend('native').get_device(0)
    assert 0 == device.index
    assert 'native:0' == device.name
