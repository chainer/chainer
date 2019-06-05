import itertools

import numpy

import chainerx


def _permutate_dtype_mapping(dtype_mapping_list):
    # Permutates in dtypes of dtype mapping.
    d = {}
    for in_dtypes, out_dtype in dtype_mapping_list:
        for in_dtypes_ in itertools.permutations(in_dtypes):
            d[in_dtypes_] = out_dtype
    return sorted(d.items())


# Used for e.g. testing power.
result_numeric_dtypes_two_arrays = _permutate_dtype_mapping([
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
])


result_comparable_dtypes_two_arrays = [
    # Bools.
    (('bool_', 'bool_'), 'bool_'),
] + result_numeric_dtypes_two_arrays


result_dtypes_two_arrays = _permutate_dtype_mapping([
    # Bools.
    (('bool_', 'bool_'), 'bool_'),
    # Bool and other.
    (('bool_', 'uint8'), 'uint8'),
    (('bool_', 'int8'), 'int8'),
    (('bool_', 'int16'), 'int16'),
    (('bool_', 'float16'), 'float16'),
    (('bool_', 'float64'), 'float64'),
]) + result_numeric_dtypes_two_arrays


result_dtypes_three_arrays = _permutate_dtype_mapping([
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
])


result_float_dtypes_array_scalar = [
    (('float16',), float, 'float16'),
    (('float32',), float, 'float32'),
    (('float64',), float, 'float64'),
    (('float16',), numpy.float64, 'float16'),
    (('float64',), numpy.float16, 'float64'),
]


result_numeric_dtypes_array_scalar = [
    # Float scalar.
    (('int8',), float, 'float32'),
    (('int16',), float, 'float32'),
    (('int32',), float, 'float32'),
    (('int64',), float, 'float32'),
    (('uint8',), float, 'float32'),
    (('int8',), numpy.float32, 'float32'),
    (('int64',), numpy.float16, 'float32'),
    (('uint8',), numpy.float64, 'float32'),
    # Int scalar.
    (('int8',), int, 'int8'),
    (('int16',), int, 'int16'),
    (('int32',), int, 'int32'),
    (('int64',), int, 'int64'),
    (('uint8',), int, 'uint8'),
    (('float16',), int, 'float16'),
    (('float32',), int, 'float32'),
    (('float64',), int, 'float64'),
    (('int16',), numpy.int16, 'int16'),
    (('uint8',), numpy.int8, 'uint8'),
    (('float64',), numpy.int8, 'float64'),
    (('float16',), numpy.int64, 'float16'),
] + result_float_dtypes_array_scalar


result_comparable_dtypes_array_scalar = [
    (('bool_',), bool, 'bool_'),
    (('bool_',), numpy.bool_, 'bool_'),
] + result_numeric_dtypes_array_scalar


result_dtypes_array_scalar = [
    # Bool scalar.
    (('bool_',), bool, 'bool_'),
    (('int8',), bool, 'int8'),
    (('int16',), bool, 'int16'),
    (('int32',), bool, 'int32'),
    (('int64',), bool, 'int64'),
    (('uint8',), bool, 'uint8'),
    (('float16',), bool, 'float16'),
    (('float32',), bool, 'float32'),
    (('float64',), bool, 'float64'),
    (('bool_',), numpy.bool_, 'bool_'),
    (('int16',), numpy.bool_, 'int16'),
    (('uint8',), numpy.bool_, 'uint8'),
    (('float32',), numpy.bool_, 'float32'),
] + result_numeric_dtypes_array_scalar


def cast_if_numpy_array(xp, array, chx_expected_dtype):
    """Casts NumPy result array to match the dtype of ChainerX's corresponding
    result.

    This function receives result arrays for both NumPy and ChainerX and only
    converts dtype of the NumPy array.
    """
    if xp is chainerx:
        assert isinstance(array, chainerx.ndarray)
        return array

    if xp is numpy:
        assert isinstance(array, (numpy.ndarray, numpy.generic))
        # Dtype conversion to allow comparing the correctnesses of the values.
        return array.astype(chx_expected_dtype, copy=False)

    assert False


def make_same_in_out_dtypes(number_of_in_params, dtypes):
    return [((dtype,) * number_of_in_params, dtype) for dtype in dtypes]
