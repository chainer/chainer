import unittest

import numpy

import chainerx

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils


class IgnoreNumpyFloatingPointError(object):

    def __enter__(self):
        self.old_settings = numpy.seterr(all='ignore')

    def __exit__(self, *args):
        numpy.seterr(**self.old_settings)


class UnaryMathTestBase(object):

    input = None

    def setup(self):
        in_dtype, = self.in_dtypes
        in_kind = numpy.dtype(in_dtype).kind

        if numpy.dtype(in_dtype).kind != 'f':
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        if in_dtype == 'float16':
            self.check_forward_options.update({'rtol': 1e-3, 'atol': 1e-3})
            self.check_backward_options.update({'rtol': 3e-3, 'atol': 3e-3})
            self.check_double_backward_options.update(
                {'rtol': 1e-2, 'atol': 1e-2})

        input = self.input
        if (in_kind == 'u'
                and isinstance(input, (int, float))
                and input < 0):
            raise unittest.SkipTest(
                'Combination of uint dtype and negative input cannot be '
                'tested')

    def generate_inputs(self):
        in_dtype, = self.in_dtypes
        if isinstance(self.input, numpy.ndarray):
            return self.input.astype(in_dtype),
        if self.input == 'random':
            return array_utils.uniform(self.shape, in_dtype),
        if isinstance(self.input, (bool, int, float)):
            return numpy.full(self.shape, self.input, dtype=in_dtype),
        assert False

    def forward_xp(self, inputs, xp):
        a, = inputs
        # This cast was introduced in order to avoid decreasing precision.
        # ex.) numpy.sqrt(x) becomes a float16 array where x is an int8 array.
        a = dtype_utils.cast_if_numpy_array(xp, a, self.out_dtype)
        with IgnoreNumpyFloatingPointError():
            y = self.func(xp, a)
        y = dtype_utils.cast_if_numpy_array(xp, y, self.out_dtype)
        return y,


class BinaryMathTestBase(object):

    def setup(self):
        in_dtype1, in_dtype2 = self.in_dtypes

        kind1 = numpy.dtype(in_dtype1).kind
        kind2 = numpy.dtype(in_dtype2).kind
        if kind1 != 'f' or kind2 != 'f':
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        if in_dtype1 == 'float16' or in_dtype2 == 'float16':
            self.check_forward_options.update({'rtol': 1e-3, 'atol': 1e-3})
            self.check_backward_options.update({'rtol': 1e-3, 'atol': 1e-3})
            self.check_double_backward_options.update(
                {'rtol': 1e-3, 'atol': 1e-3})

    def generate_inputs(self):
        in_dtype1, in_dtype2 = self.in_dtypes
        in_shape1, in_shape2 = self.in_shapes
        if self.input_lhs == 'random':
            a = array_utils.uniform(in_shape1, in_dtype1)
        elif isinstance(self.input_lhs, (bool, int, float)):
            a = numpy.full(in_shape1, self.input_lhs, dtype=in_dtype1)
        else:
            assert False
        if self.input_rhs == 'random':
            b = array_utils.uniform(in_shape2, in_dtype2)
        elif isinstance(self.input_rhs, (bool, int, float)):
            b = numpy.full(in_shape2, self.input_rhs, dtype=in_dtype2)
        else:
            assert False
        return a, b

    def forward_xp(self, inputs, xp):
        a, b = inputs
        # This cast was introduced in order to avoid decreasing precision.
        # ex.) x / y becomes a float16 array where x and y are an int8 arrays.
        a = dtype_utils.cast_if_numpy_array(xp, a, self.out_dtype)
        b = dtype_utils.cast_if_numpy_array(xp, b, self.out_dtype)
        with IgnoreNumpyFloatingPointError():
            y = self.func(xp, a, b)
        y = dtype_utils.cast_if_numpy_array(xp, y, self.out_dtype)
        return y,


class InplaceUnaryMathTestBase(UnaryMathTestBase):

    skip_backward_test = True
    skip_double_backward_test = True

    def forward_xp(self, inputs, xp):
        a, = inputs
        if xp is chainerx:
            a_ = a.as_grad_stopped().copy()
        else:
            a_ = a.copy()
        with IgnoreNumpyFloatingPointError():
            ret = self.func(xp, a_)
        assert ret is None  # func should not return anything
        return a_,


class InplaceBinaryMathTestBase(BinaryMathTestBase):

    skip_backward_test = True
    skip_double_backward_test = True

    def forward_xp(self, inputs, xp):
        a, b = inputs
        b = dtype_utils.cast_if_numpy_array(xp, b, a.dtype)
        if xp is chainerx:
            a_ = a.as_grad_stopped().copy()
            b_ = b.as_grad_stopped()
        else:
            a_ = a.copy()
            b_ = b
        with IgnoreNumpyFloatingPointError():
            ret = self.func(xp, a_, b_)
        assert ret is None  # func should not return anything
        return a_,


def _convert_numpy_scalar(scalar, dtype):
    # Implicit casting in NumPy's multiply depends on the 'casting' argument,
    # which is not yet supported (ChainerX always casts).
    # Therefore, we explicitly cast the scalar to the dtype of the ndarray
    # before the multiplication for NumPy.
    return numpy.dtype(dtype).type(scalar)


class MathScalarTestBase(UnaryMathTestBase):

    def func(self, xp, a):
        scalar = self.scalar_type(self.scalar_value)
        return self.func_scalar(xp, a, scalar)


class InplaceMathScalarTestBase(InplaceUnaryMathTestBase):

    def func(self, xp, a):
        scalar = self.scalar_type(self.scalar_value)
        if xp is numpy:
            # This cast is to avoid TypeError in the following case
            #     a: uint8 0-dim numpy.ndarray
            #     scalar: int
            in_dtype, = self.in_dtypes
            scalar = _convert_numpy_scalar(scalar, in_dtype)
        return self.func_scalar(xp, a, scalar)


def _permutate_shapes(shapes_list):
    # Permutates input shapes
    permutated_shapes_list = []
    for in_shape1, in_shape2 in shapes_list:
        permutated_shapes_list.append((in_shape1, in_shape2))
        permutated_shapes_list.append((in_shape2, in_shape1))
    return list(set(permutated_shapes_list))


shapes_combination_inplace_binary = [
    # Same shapes
    ((1,), (1,)),
    ((3, 4), (3, 4)),
    # Broadcast
    ((10,), (1,)),
    ((3, 4), (3, 1)),
    ((3, 4), (1, 4)),
    ((3, 4), (4,)),
    ((3, 4), (1, 1)),
    ((3, 4), (1,)),
    ((2, 3, 4), (1, 1, 1)),
    # 0-dim shape
    ((), ()),
    ((1,), ()),
    ((3,), ()),
    ((2, 3), ()),
    # 0-size shape
    ((0,), (0,)),
    ((0,), (1,)),
    ((0,), ()),
    ((2, 0, 3), (2, 0, 3)),
    # TODO(imanishi): Fix strides
    # ((2, 0, 3), (0, 1)),
]


shapes_combination_binary = _permutate_shapes([
    # Broadcast
    ((3, 1), (1, 4)),
    ((2, 1, 4), (3, 1)),
    # 0-size shape
    # TODO(imanishi): Fix strides
    # ((0, 1), (0, 1, 0)),
]) + _permutate_shapes(shapes_combination_inplace_binary)


# An association list that associates a dtype to the type which ChainerX's
# real-valued functions should return.
in_out_float_dtypes_math_functions = [
    # Float.
    (('float16',), 'float16'),
    (('float32',), 'float32'),
    (('float64',), 'float64'),
]


in_out_dtypes_math_functions = in_out_float_dtypes_math_functions + [
    # Signed int.
    (('int8',), 'float32'),
    (('int16',), 'float32'),
    (('int32',), 'float32'),
    (('int64',), 'float32'),
    # Unsigned int.
    (('uint8',), 'float32'),
    # Bool.
    (('bool_',), 'float32'),
]


in_out_dtypes_math_binary_functions = dtype_utils._permutate_dtype_mapping([
    # integer mixed
    (('int8', 'int16'), 'float32'),
    (('int8', 'int32'), 'float32'),
    (('int8', 'int64'), 'float32'),
    (('int8', 'uint8'), 'float32'),
    (('int16', 'int32'), 'float32'),
    (('int16', 'int64'), 'float32'),
    (('int16', 'uint8'), 'float32'),
    (('int32', 'int64'), 'float32'),
    (('int32', 'uint8'), 'float32'),
    (('int64', 'uint8'), 'float32'),
    # integer float mixed
    (('int8', 'float16'), 'float16'),
    (('int8', 'float32'), 'float32'),
    (('int8', 'float64'), 'float64'),
    (('int16', 'float16'), 'float16'),
    (('int16', 'float32'), 'float32'),
    (('int16', 'float64'), 'float64'),
    (('int32', 'float16'), 'float16'),
    (('int32', 'float32'), 'float32'),
    (('int32', 'float64'), 'float64'),
    (('int64', 'float16'), 'float16'),
    (('int64', 'float32'), 'float32'),
    (('int64', 'float64'), 'float64'),
    (('uint8', 'float16'), 'float16'),
    (('uint8', 'float32'), 'float32'),
    (('uint8', 'float64'), 'float64'),
    # float mixed
    (('float16', 'float32'), 'float32'),
    (('float16', 'float64'), 'float64'),
    (('float32', 'float64'), 'float64'),
])
