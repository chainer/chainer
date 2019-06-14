import chainerx


def test_core():
    assert chainerx.__name__ == 'chainerx'


def test_is_available():
    assert chainerx.is_available()
