import xchainer


def test_inheritance():
    assert xchainer.NativeBackend.__bases__[0] == xchainer.Backend


def test_creation():
    xchainer.NativeBackend()
