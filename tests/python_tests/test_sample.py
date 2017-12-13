import xchainer


def test_sample():
    assert xchainer.__name__ == 'xchainer'
    assert xchainer.hello() == 'world'
