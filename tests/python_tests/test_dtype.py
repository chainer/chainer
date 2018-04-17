import numpy
import pytest

import xchainer
import xchainer.testing


_dtypes_data = [
    ('bool_', 'bool', '?', 1),
    ('int8', 'int8', 'b', 1),
    ('int16', 'int16', 'h', 2),
    ('int32', 'int32', 'i', 4),
    ('int64', 'int64', 'q', 8),
    ('uint8', 'uint8', 'B', 1),
    ('float32', 'float32', 'f', 4),
    ('float64', 'float64', 'd', 8),
]


def test_py_types():
    assert xchainer.bool is bool
    assert xchainer.int is int
    assert xchainer.float is float


def test_dtype_from_python_type():
    assert xchainer.dtype('bool') == xchainer.bool_
    assert xchainer.dtype('int') == xchainer.int64
    assert xchainer.dtype('float') == xchainer.float64
    assert xchainer.dtype(bool) == xchainer.bool_
    assert xchainer.dtype(int) == xchainer.int64
    assert xchainer.dtype(float) == xchainer.float64


@xchainer.testing.parametrize_dtype_specifier('dtype_spec', with_xchainer_dtypes=False)
def test_dtype_from_args(dtype_spec):
    assert xchainer.dtype(dtype_spec).name == numpy.dtype(dtype_spec).name


def test_dtypes_covered(dtype):
    # This test ensures _dtypes_data covers all dtypes
    assert any(tup[0] == dtype for tup in _dtypes_data), 'Not all dtypes are covered in _dtypes_data in dtypes test.'


@pytest.mark.parametrize("dtype_symbol,name,char,itemsize", _dtypes_data)
def test_dtypes(dtype_symbol, name, char, itemsize):
    dtype = getattr(xchainer, dtype_symbol)
    assert isinstance(dtype, xchainer.dtype)
    assert dtype.name == name
    assert dtype.char == char
    assert dtype.itemsize == itemsize
    assert xchainer.dtype(name) == dtype
    assert xchainer.dtype(char) == dtype
    assert xchainer.dtype(dtype) == dtype
    # From NumPy dtypes
    assert xchainer.dtype(numpy.dtype(dtype_symbol)) == dtype


def test_eq():
    assert xchainer.int8 == xchainer.int8
    assert xchainer.dtype('int8') == xchainer.int8
    assert xchainer.dtype(xchainer.int8) == xchainer.int8
    assert not 8 == xchainer.int8
    assert not xchainer.int8 == 8
    assert not 'int8' == xchainer.int8
    assert not xchainer.int8 == 'int8'


def test_ne():
    assert xchainer.int32 != xchainer.int8
    assert xchainer.dtype('int32') != xchainer.int8
    assert xchainer.dtype(xchainer.int32) != xchainer.int8
    assert 32 != xchainer.int32
    assert xchainer.int8 != 32
    assert 'int32' != xchainer.int32
    assert xchainer.int8 != 'int32'


def test_implicity_convertible():
    xchainer.zeros(shape=(2, 3), dtype='int32')
