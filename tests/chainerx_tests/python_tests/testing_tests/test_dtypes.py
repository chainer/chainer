import numpy

import chainerx
import chainerx.testing


def test_float_dtypes():
    # float_dtypes must be a subset of all_dtypes
    assert all(
        dtype in chainerx.testing.all_dtypes
        for dtype in chainerx.testing.float_dtypes)

    # Check dtype kind
    for dtype in chainerx.testing.all_dtypes:
        is_float = dtype in chainerx.testing.float_dtypes
        assert is_float == (numpy.dtype(
            getattr(numpy, dtype)).kind in ('f', 'c'))


def test_signed_dtypes():
    # signe_dtypes must be a subset of all_dtypes
    assert all(
        dtype in chainerx.testing.all_dtypes
        for dtype in chainerx.testing.signed_dtypes)

    # Check dtype kind
    for dtype in chainerx.testing.all_dtypes:
        is_signed = dtype in chainerx.testing.signed_dtypes
        assert is_signed == (numpy.dtype(
            getattr(numpy, dtype)).kind in ('i', 'f', 'c'))


@chainerx.testing.parametrize_dtype_specifier('spec')
def test_parametrize_dtype_specifier(spec):
    assert numpy.dtype(spec).type.__name__ in chainerx.testing.all_dtypes


@chainerx.testing.parametrize_dtype_specifier(
    'spec', dtypes=['int32', 'float64'])
def test_parametrize_dtype_specifier_with_dtypes(spec):
    assert chainerx.dtype(spec).name in ('int32', 'float64')


@chainerx.testing.parametrize_dtype_specifier(
    'spec', dtypes=[], additional_args=('foo',))
def test_parametrize_dtype_specifier_with_additional_args(spec):
    assert spec == 'foo'
