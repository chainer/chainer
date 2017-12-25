import pytest

import xchainer
from xchainer import Dtype


@pytest.fixture
def inputs(dtype_data):
    name = dtype_data['name']
    char = dtype_data['char']
    itemsize = dtype_data['itemsize']
    return name, char, itemsize


def test_inti_eq(inputs):
    name, char, itemsize = inputs
    dtype = getattr(xchainer, name)

    assert dtype.name == name
    assert dtype.char == char
    assert dtype.itemsize == itemsize
    assert Dtype(name) == dtype
    assert Dtype(char) == dtype
