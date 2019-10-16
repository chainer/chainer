import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import dtype_utils
from chainerx_tests import math_utils


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0.5),
    numpy.asarray(-1.2),
    numpy.asarray(10.9),
    numpy.asarray(float('inf')),
    numpy.asarray(-float('inf')),
    numpy.asarray(float('nan')),
    numpy.full((), 2.1),
    numpy.full((0,), 2),
    numpy.full((2, 3), 2.6),
    numpy.full((1, 1), 1.01),
    numpy.full((1, 1), 1.99),
])
@pytest.mark.parametrize('dtypes', math_utils.in_out_dtypes_math_functions)
@pytest.mark.parametrize('func', [
    lambda xp, a: xp.ceil(a),
    lambda xp, a: xp.floor(a)
])
def test_rounding_routines(func, xp, device, input, dtypes):
    (in_dtype, ), out_dtype = dtypes
    a = xp.array(input.astype(in_dtype))
    a = func(xp, a)
    a = dtype_utils.cast_if_numpy_array(xp, a, out_dtype)
    return a
