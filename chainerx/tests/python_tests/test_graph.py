import chainerx


def test_anygraph():
    assert hasattr(chainerx, 'anygraph')
