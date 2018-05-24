import math

import numpy
import pytest

import xchainer
import xchainer.testing


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
def test_dtype_from_specifier(dtype_spec):
    assert xchainer.dtype(dtype_spec).name == numpy.dtype(dtype_spec).name


@pytest.mark.parametrize('dtype_symbol', xchainer.testing.all_dtypes)
def test_dtypes(dtype_symbol):
    dtype = getattr(xchainer, dtype_symbol)
    numpy_dtype = numpy.dtype(dtype_symbol)
    assert isinstance(dtype, xchainer.dtype)
    assert dtype.name == numpy_dtype.name
    assert dtype.char == numpy_dtype.char
    assert dtype.itemsize == numpy_dtype.itemsize
    assert dtype.kind == numpy_dtype.kind
    assert dtype.byteorder == numpy_dtype.byteorder
    assert dtype.str == numpy_dtype.str
    assert dtype.num == numpy_dtype.num
    assert xchainer.dtype(dtype.name) == dtype
    assert xchainer.dtype(dtype.char) == dtype
    assert xchainer.dtype(dtype) == dtype
    # From NumPy dtypes
    assert xchainer.dtype(numpy_dtype) == dtype


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


@xchainer.testing.parametrize_dtype_specifier('dtype_spec')
@pytest.mark.parametrize('value', [
    -2,
    1,
    -1.5,
    2.3,
    True,
    False,
    numpy.array(1),
    float('inf'),
    float('nan'),
])
def test_type(dtype_spec, value):
    expected = xchainer.Scalar(value, dtype_spec)
    actual = xchainer.dtype(dtype_spec).type(value)
    assert isinstance(actual, xchainer.Scalar)
    assert actual.dtype == xchainer.dtype(dtype_spec)
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert expected == actual
