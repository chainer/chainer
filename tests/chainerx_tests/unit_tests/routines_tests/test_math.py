import chainer
import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import dtype_utils
from chainerx_tests import math_utils
from chainerx_tests import op_utils


_in_out_dtypes_arithmetic_invalid = [
    (('bool_', 'bool_'), 'bool_'),
    (('bool_', 'int8'), 'int8'),
    (('bool_', 'int16'), 'int16'),
    (('bool_', 'int32'), 'int32'),
    (('bool_', 'int64'), 'int64'),
    (('bool_', 'uint8'), 'uint8'),
    (('bool_', 'float16'), 'float16'),
    (('bool_', 'float32'), 'float32'),
    (('bool_', 'float64'), 'float64'),
    (('int8', 'bool_'), 'int8'),
    (('int16', 'bool_'), 'int16'),
    (('int32', 'bool_'), 'int32'),
    (('int64', 'bool_'), 'int64'),
    (('uint8', 'bool_'), 'uint8'),
    (('float16', 'bool_'), 'float16'),
    (('float32', 'bool_'), 'float32'),
    (('float64', 'bool_'), 'float64'),
]


_in_out_dtypes_arithmetic = [
    dtypes for dtypes in dtype_utils.result_dtypes_two_arrays
    if dtypes not in _in_out_dtypes_arithmetic_invalid
]


_in_out_dtypes_inplace_arithmetic_invalid = [
    ((t1, t2), t3) for (t1, t2), t3 in _in_out_dtypes_arithmetic
    if (numpy.dtype(t1).kind != 'f' and numpy.dtype(t2).kind == 'f')
] + _in_out_dtypes_arithmetic_invalid


_in_out_dtypes_inplace_arithmetic = [
    dtypes for dtypes in dtype_utils.result_dtypes_two_arrays
    if dtypes not in _in_out_dtypes_inplace_arithmetic_invalid
]


_in_out_dtypes_array_int_scalar = [
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
]


_in_out_dtypes_int_array_float_scalar = [
    # Int arrays and float scalars.
    (('int8',), float, 'float32'),
    (('int16',), float, 'float32'),
    (('int32',), float, 'float32'),
    (('int64',), float, 'float32'),
    (('uint8',), float, 'float32'),
    (('int8',), numpy.float32, 'float32'),
    (('int64',), numpy.float16, 'float32'),
    (('uint8',), numpy.float64, 'float32'),
]


_in_out_dtypes_float_array_float_scalar = [
    # Float arrays and flaot scalars.
    (('float16',), float, 'float16'),
    (('float32',), float, 'float32'),
    (('float64',), float, 'float64'),
    (('float64',), float, 'float64'),
    (('float16',), numpy.float64, 'float16'),
    (('float64',), numpy.float16, 'float64'),
]


_in_out_dtypes_arithmetic_scalar = (
    _in_out_dtypes_array_int_scalar
    + _in_out_dtypes_int_array_float_scalar
    + _in_out_dtypes_float_array_float_scalar)


_in_out_dtypes_inplace_arithmetic_scalar = (
    _in_out_dtypes_array_int_scalar
    + _in_out_dtypes_float_array_float_scalar)


_in_out_dtypes_float_arithmetic_scalar = (
    _in_out_dtypes_int_array_float_scalar
    + _in_out_dtypes_float_array_float_scalar)


_in_out_dtypes_inplace_float_arithmetic_scalar = (
    _in_out_dtypes_float_array_float_scalar)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Differentiable
    chainer.testing.product({
        'input': [
            numpy.asarray(0.),
            numpy.asarray(-1.),
            numpy.asarray(1.),
            numpy.asarray(10.),
            numpy.full((), 2.),
            numpy.full((0,), 2.),
            numpy.full((2, 3), 2.)
        ]})
    +
    # Nondifferentiable
    chainer.testing.product({
        'input': [
            numpy.asarray(float('inf')),
            numpy.asarray(float('nan')),
        ],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
@pytest.mark.parametrize('contiguous', [None, 'C'])
class TestSigmoid(op_utils.NumpyOpTest):

    def setup(self, contiguous, float_dtype):
        self.dtype = float_dtype
        self.contiguous = contiguous
        self.check_forward_options = {'atol': 5e-3, 'rtol': 5e-3}

        if float_dtype == 'float16':
            self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-2}

    def generate_inputs(self):
        return self.input,

    def forward_xp(self, inputs, xp):
        if xp is numpy:
            return 1 / (1 + numpy.exp(-inputs[0])),
        return xp.sigmoid(inputs[0]),


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0), numpy.asarray(-1), numpy.asarray(
        10), numpy.asarray(float('inf')), numpy.asarray(-float('inf')),
    numpy.asarray(float('nan')), numpy.full(
        (), 2), numpy.full((0,), 2), numpy.full((2, 3), 2)
])
def test_isnan(xp, device, input, dtype):
    a = xp.array(input.astype(dtype))
    return xp.isnan(a)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0), numpy.asarray(-1), numpy.asarray(
        10), numpy.asarray(float('inf')), numpy.asarray(-float('inf')),
    numpy.asarray(float('nan')), numpy.full(
        (), 2), numpy.full((0,), 2), numpy.full((2, 3), 2)
])
def test_isinf(xp, device, input, dtype):
    a = xp.array(input.astype(dtype))
    return xp.isinf(a)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0), numpy.asarray(-1), numpy.asarray(
        10), numpy.asarray(float('inf')), numpy.asarray(-float('inf')),
    numpy.asarray(float('nan')), numpy.full(
        (), 2), numpy.full((0,), 2), numpy.full((2, 3), 2)
])
def test_isfinite(xp, device, input, dtype):
    a = xp.array(input.astype(dtype))
    return xp.isfinite(a)
