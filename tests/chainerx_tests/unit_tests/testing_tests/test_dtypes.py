import numpy

import chainerx
import chainerx.testing


def test_dtype_lists():
    # Each dtype list must be a subset of all_dtypes.
    dtype_lists = [
        chainerx.testing.all_dtypes,
        chainerx.testing.float_dtypes,
        chainerx.testing.signed_dtypes,
        chainerx.testing.unsigned_dtypes,
        chainerx.testing.integral_dtypes,
        chainerx.testing.nonfloat_dtypes,
    ]
    for dtype_list in dtype_lists:
        assert isinstance(dtype_list, tuple)
        assert all([
            dtype in chainerx.testing.all_dtypes
            for dtype in dtype_list])

    # Check dtype kind
    for dtype in chainerx.testing.all_dtypes:
        is_float = dtype in chainerx.testing.float_dtypes
        assert is_float == (numpy.dtype(dtype).kind in ('f', 'c'))

        is_signed = dtype in chainerx.testing.signed_dtypes
        assert is_signed == (numpy.dtype(dtype).kind in ('i', 'f', 'c'))

        is_unsigned = dtype in chainerx.testing.unsigned_dtypes
        assert is_unsigned == (numpy.dtype(dtype).kind == 'u')

        is_integral = dtype in chainerx.testing.integral_dtypes
        assert is_integral == (numpy.dtype(dtype).kind in ('i', 'u'))

        is_nonfloat = dtype in chainerx.testing.nonfloat_dtypes
        assert is_nonfloat == (numpy.dtype(dtype).kind != 'f')


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
