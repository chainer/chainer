import chainer
import numpy

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import op_utils


class IgnoreNumpyFloatingPointError(object):

    def __enter__(self):
        self.old_settings = numpy.seterr(all='ignore')

    def __exit__(self, *args):
        numpy.seterr(**self.old_settings)


class UnaryMathTestBase(object):

    def setup(self):
        in_dtype, = self.in_dtypes

        if numpy.dtype(in_dtype).kind != 'f':
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        if in_dtype == 'float16':
            self.check_forward_options.update({'rtol': 1e-3, 'atol': 1e-3})
            self.check_backward_options.update({'rtol': 1e-3, 'atol': 1e-3})
            self.check_double_backward_options.update(
                {'rtol': 1e-2, 'atol': 1e-2})

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


_in_out_float_dtypes_math_functions = [
    # Float.
    (('float16',), 'float16'),
    (('float32',), 'float32'),
    (('float64',), 'float64'),
]


_in_out_dtypes_math_functions = _in_out_float_dtypes_math_functions + [
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


@op_utils.op_test(['native:0', 'cuda:0'])
class TestLeakyRelu(op_utils.OpTest):

    slope = 0.2

    def setup(self, shape, float_dtype):
        self.dtype = float_dtype
        self.shape = shape

        if float_dtype == 'float16':
            self.check_forward_options.update({'atol': 1e-4, 'rtol': 1e-3})
            self.check_backward_options.update({'atol': 1e-2, 'rtol': 5e-2})
            self.check_double_backward_options.update(
                {'atol': 1e-2, 'rtol': 5e-2})

    def generate_inputs(self):
        dtype = self.dtype
        shape = self.shape
        x = array_utils.create_dummy_ndarray(numpy, shape, dtype)
        return x,

    def forward_chainerx(self, inputs):
        x, = inputs
        y = chainerx.leaky_relu(x, self.slope)
        return y,

    def forward_expected(self, inputs):
        x, = inputs
        expected = numpy.where(x >= 0, x, x * self.slope)
        return expected.astype(self.dtype),


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
        'input': [-2, 2],
        'contiguous': [None, 'C'],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
        'input': [0, float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestRelu(UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        if xp is numpy:
            return numpy.maximum(a, 0)
        return xp.relu(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
        'input': [0, -1, 1, -2, 2, 10],
        'contiguous': [None, 'C'],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
        'input': [0, float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestSigmoid(UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        if xp is numpy:
            return numpy.asarray(
                numpy.reciprocal(1 + numpy.exp(-a))).astype(a.dtype)
        return xp.sigmoid(a)
