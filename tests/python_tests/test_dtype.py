import pytest

import xchainer


@pytest.fixture
def dtype_init_inputs(dtype_data):
    name = dtype_data['name']
    char = dtype_data['char']
    itemsize = dtype_data['itemsize']
    return name, char, itemsize


def test_inti_eq(dtype_init_inputs):
    name, char, itemsize = dtype_init_inputs
    dtype = getattr(xchainer, name)

    assert dtype.name == name
    assert dtype.char == char
    assert dtype.itemsize == itemsize
    assert xchainer.Dtype(name) == dtype
    assert xchainer.Dtype(char) == dtype
