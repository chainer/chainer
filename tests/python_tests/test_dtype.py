import pytest

import xchainer


_dtypes_data = [
    (xchainer.bool, 'bool', '?', 1),
    (xchainer.int8, 'int8', 'b', 1),
    (xchainer.int16, 'int16', 'h', 2),
    (xchainer.int32, 'int32', 'i', 4),
    (xchainer.int64, 'int64', 'q', 8),
    (xchainer.uint8, 'uint8', 'B', 1),
    (xchainer.float32, 'float32', 'f', 4),
    (xchainer.float64, 'float64', 'd', 8),
]


def test_dtypes_covered(dtype):
    # This test ensures _dtypes_data covers all dtypes
    assert any(tup[0] == dtype for tup in _dtypes_data), 'Not all dtypes are covered in _dtypes_data in dtypes test.'


@pytest.mark.parametrize("dtype,name,char,itemsize", _dtypes_data)
def test_dtypes(dtype, name, char, itemsize):
    assert dtype == getattr(xchainer, name)
    assert dtype.name == name
    assert dtype.char == char
    assert dtype.itemsize == itemsize
    assert xchainer.dtype(name) == dtype
    assert xchainer.dtype(char) == dtype
    assert xchainer.dtype(dtype) == dtype


@pytest.mark.parametrize("dtype,name,char,itemsize", _dtypes_data)
def test_implicitly_convertible(dtype, name, char, itemsize):
    assert dtype == dtype
    assert dtype == name
    assert dtype == char


def test_eq():
    assert xchainer.int8 == xchainer.int8
    assert xchainer.dtype('int8') == xchainer.int8
    assert xchainer.dtype(xchainer.int8) == xchainer.int8
    assert not 8 == xchainer.int8
    assert not xchainer.int8 == 8


def test_ne():
    assert xchainer.int32 != xchainer.int8
    assert xchainer.dtype('int32') != xchainer.int8
    assert xchainer.dtype(xchainer.int32) != xchainer.int8
    assert 32 != xchainer.int32
    assert xchainer.int8 != 32
