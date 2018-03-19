import xchainer


def test_newaxis():
    assert xchainer.newaxis is None


def test_broadcastable():
    assert xchainer.broadcastable is xchainer.broadcastable
