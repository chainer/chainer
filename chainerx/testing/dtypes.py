import numpy
import pytest

import chainerx


float_dtypes = (
    'float16',
    'float32',
    'float64',
)

signed_dtypes = (
    'int8',
    'int16',
    'int32',
    'int64',
    'float16',
    'float32',
    'float64',
)


unsigned_dtypes = (
    'uint8',
)


nonfloat_dtypes = (
    'bool_',
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
)


numeric_dtypes = signed_dtypes + unsigned_dtypes


all_dtypes = ('bool_',) + numeric_dtypes


_result_dtypes_two_arrays = [
    # Bools.
    (('bool_', 'bool_'), 'bool'),
    # Floats.
    (('float16', 'float16'), 'float16'),
    (('float32', 'float32'), 'float32'),
    (('float64', 'float64'), 'float64'),
    (('float32', 'float16'), 'float32'),
    (('float64', 'float16'), 'float64'),
    (('float64', 'float32'), 'float64'),
    # Signed ints.
    (('int8', 'int8'), 'int8'),
    (('int8', 'int16'), 'int16'),
    (('int8', 'int32'), 'int32'),
    (('int8', 'int64'), 'int64'),
    (('int16', 'int16'), 'int16'),
    (('int32', 'int32'), 'int32'),
    (('int64', 'int64'), 'int64'),
    (('int16', 'int32'), 'int32'),
    (('int16', 'int64'), 'int64'),
    (('int32', 'int64'), 'int64'),
    # Unsigned ints.
    (('uint8', 'uint8'), 'uint8'),
    # Signed int and unsigned int.
    (('uint8', 'int8'), 'int16'),
    (('uint8', 'int16'), 'int16'),
    (('uint8', 'int32'), 'int32'),
    # Int and float.
    (('int8', 'float16'), 'float16'),
    (('uint8', 'float16'), 'float16'),
    (('int16', 'float32'), 'float32'),
    (('int32', 'float32'), 'float32'),
    (('int64', 'float32'), 'float32'),
    # Bool and other.
    (('bool_', 'uint8'), 'uint8'),
    (('bool_', 'int8'), 'int8'),
    (('bool_', 'int16'), 'int16'),
    (('bool_', 'float16'), 'float16'),
    (('bool_', 'float64'), 'float64'),
]


_result_dtypes_three_arrays = [
    # Signed ints.
    (('int32', 'int32', 'int32'), 'int32'),
    (('int8', 'int8', 'int32'), 'int32'),
    (('int8', 'int16', 'int32'), 'int32'),
    (('int8', 'int32', 'int32'), 'int32'),
    (('int8', 'int64', 'int32'), 'int64'),
    # Unsigned ints.
    (('uint8', 'uint8', 'uint8'), 'uint8'),
    (('uint8', 'uint8', 'int8'), 'int16'),
    (('uint8', 'int8', 'int8'), 'int16'),
    (('uint8', 'int8', 'int16'), 'int16'),
    (('uint8', 'uint8', 'int16'), 'int16'),
    # Float and signed int.
    (('float16', 'int8', 'int8'), 'float16'),
    (('float16', 'int32', 'int64'), 'float16'),
    (('float16', 'float32', 'int64'), 'float32'),
    # Float and unsigned int.
    (('float16', 'int8', 'uint8'), 'float16'),
    (('float16', 'int32', 'uint8'), 'float16'),
    (('float16', 'float32', 'uint8'), 'float32'),
    # Bool and other.
    (('bool_', 'uint8', 'uint8'), 'uint8'),
    (('bool_', 'bool_', 'uint8'), 'uint8'),
    (('bool_', 'int8', 'uint8'), 'int16'),
    (('bool_', 'bool_', 'int32'), 'int32'),
    (('bool_', 'float16', 'float32'), 'float32'),
    (('bool_', 'bool_', 'float64'), 'float64'),
]


def parametrize_dtype_specifier(argname, dtypes=None, additional_args=None):
    """Parametrizes a test with various arguments that can be used as dtypes.

    Args:
         argname(str): Argument name to pass the value that can be used as a
             dtype.
         dtypes(list of strs): List of dtype names.
         additional_args(tuple of list): Additional values to be included.
    """
    if dtypes is None:
        dtypes = all_dtypes
    assert isinstance(argname, str)
    assert isinstance(dtypes, (tuple, list))
    assert all(isinstance(dt, str) for dt in dtypes)
    lst = []

    # dtype names
    lst += list(dtypes)
    # numpy dtypes
    lst += [numpy.dtype(dt) for dt in dtypes]
    # char codes
    lst += [chainerx.dtype(dt).char for dt in dtypes]
    # User-specified args
    if additional_args is not None:
        assert isinstance(additional_args, (tuple, list))
        lst += list(additional_args)

    return pytest.mark.parametrize(argname, lst)
