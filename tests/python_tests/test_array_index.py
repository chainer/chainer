import pytest

import xchainer


def test_init():
    xchainer.ArrayIndex(1)
    xchainer.ArrayIndex(slice(1, 2, 1))
    xchainer.ArrayIndex(None)
    xchainer.ArrayIndex(xchainer.newaxis)
