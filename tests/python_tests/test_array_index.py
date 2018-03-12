import pytest

import xchainer


def test_init():
    xchainer.ArrayIndex(1)
    xchainer.ArrayIndex(slice(1, 2, 1))
    xchainer.ArrayIndex(None)
    xchainer.ArrayIndex(xchainer.newaxis)

    a = xchainer.empty((2,3), xchainer.float32)
    b = a[1, 1::2]
    print(a)
    print(b)
