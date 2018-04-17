import numpy

import xchainer
import xchainer.testing


def test_float_dtypes():
    # float_dtypes must be a subset of all_dtypes
    assert all(dtype in xchainer.testing.all_dtypes for dtype in xchainer.testing.float_dtypes)

    # Check dtype kind
    for dtype in xchainer.testing.all_dtypes:
        is_float = dtype in xchainer.testing.float_dtypes
        assert is_float == (numpy.dtype(getattr(numpy, dtype)).kind in ('f', 'c'))


def test_signed_dtypes():
    # signe_dtypes must be a subset of all_dtypes
    assert all(dtype in xchainer.testing.all_dtypes for dtype in xchainer.testing.signed_dtypes)

    # Check dtype kind
    for dtype in xchainer.testing.all_dtypes:
        is_signed = dtype in xchainer.testing.signed_dtypes
        assert is_signed == (numpy.dtype(getattr(numpy, dtype)).kind in ('i', 'f', 'c'))


@xchainer.testing.parametrize_dtype_specifier('spec', with_xchainer_dtypes=False)
def test_parametrize_dtype_specifier(spec):
    assert numpy.dtype(spec).type.__name__ in xchainer.testing.all_dtypes


@xchainer.testing.parametrize_dtype_specifier('spec', dtypes=['int32', 'float64'])
def test_parametrize_dtype_specifier_with_dtypes(spec):
    assert xchainer.dtype(spec).name in ('int32', 'float64')


@xchainer.testing.parametrize_dtype_specifier('spec', dtypes=[], additional_args=('foo',))
def test_parametrize_dtype_specifier_with_additional_args(spec):
    assert spec == 'foo'
