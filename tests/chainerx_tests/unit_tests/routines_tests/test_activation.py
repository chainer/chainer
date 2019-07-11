import random
import chainer
import numpy

from chainer import utils
from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import op_utils


# A special parameter object used to represent an unspecified argument.
class Unspecified(object):
    pass


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
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
        'input': [-2, 2],
        'contiguous': [None, 'C'],
        'alpha_range': [(-2.0, 0.0), 0.0, (0.0, 2.0), Unspecified],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
        'input': [0, float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
        'alpha_range': [(-2.0, 0.0), 0.0, (0.0, 2.0), Unspecified],
    })
))
class TestClippedRelu(UnaryMathTestBase, op_utils.NumpyOpTest):

    z = 0.75

    def func(self, xp, a):
        dtype = self.out_dtype
        if xp is numpy:
            y = utils.force_array(a.clip(0, self.z))
            return numpy.asarray(y.astype(dtype))
        return xp.clipped_relu(a, self.z)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape,axis': [
            ((5, 4), 0),
            ((5, 4), 1),
            ((5, 4), -1),
            ((5, 4), -2),
            ((5, 4, 3, 2), 0),
            ((5, 4, 3, 2), 1),
            ((5, 4, 3, 2), 2),
            ((5, 4, 3, 2), 3),
            ((5, 4, 3, 2), -1),
            ((5, 4, 3, 2), -2),
            ((5, 4, 3, 2), -3),
            ((5, 4, 3, 2), -4),
        ],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
    })
))
class TestCRelu(UnaryMathTestBase, op_utils.NumpyOpTest):

    check_numpy_strides_compliance = False
    dodge_nondifferentiable = True

    def generate_inputs(self):
        in_dtype, = self.in_dtypes
        a = array_utils.uniform(self.shape, in_dtype)
        return a,

    def func(self, xp, a):
        if xp is numpy:
            expected_former = numpy.maximum(a, 0)
            expected_latter = numpy.maximum(-a, 0)
            expected = numpy.concatenate(
                (expected_former, expected_latter), axis=self.axis)
            return expected
        return xp.crelu(a, self.axis)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
        'input': [-2, 2],
        'contiguous': [None, 'C'],
        'alpha_range': [(-2.0, 0.0), 0.0, (0.0, 2.0), Unspecified],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
        'input': [0, float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
        'alpha_range': [(-2.0, 0.0), 0.0, (0.0, 2.0), Unspecified],
    })
))
class TestElu(UnaryMathTestBase, op_utils.NumpyOpTest):

    def setup(self):
        in_dtype, = self.in_dtypes
        if isinstance(self.alpha_range, tuple):
            l, u = self.alpha_range
            self.alpha = random.uniform(l, u)
        elif self.alpha_range is Unspecified:
            self.alpha = 1.0
        else:
            self.alpha = self.alpha_range

        if numpy.dtype(in_dtype).kind != 'f':
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        if in_dtype == 'float16':
            self.check_forward_options.update({'rtol': 1e-3, 'atol': 1e-3})
            self.check_backward_options.update({'rtol': 2e-3, 'atol': 2e-3})
            self.check_double_backward_options.update(
                {'rtol': 1e-2, 'atol': 1e-2})

    def func(self, xp, a):
        if xp is numpy:
            y = a.copy()
            negzero_indices = y <= 0
            y[negzero_indices] = self.alpha * numpy.expm1(y[negzero_indices])
            return y
        elif self.alpha_range is Unspecified:
            return xp.elu(a)
        else:
            return xp.elu(a, self.alpha)


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
class TestLeakyRelu(UnaryMathTestBase, op_utils.NumpyOpTest):

    slope = 0.2
    check_numpy_strides_compliance = False

    def func(self, xp, a):
        if xp is numpy:
            expected = numpy.where(a >= 0, a, a * self.slope)
            return expected
        return xp.leaky_relu(a, self.slope)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': _in_out_dtypes_math_functions,
        'input': [-2, 2],
        'contiguous': [None, 'C'],
        'beta_range': [(-2.0, -1.0), (1.0, 2.0), Unspecified],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': _in_out_float_dtypes_math_functions,
        'input': [0, float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
        'beta_range': [(-2.0, -1.0), (1.0, 2.0), Unspecified],
    })
))
class TestSoftplus(UnaryMathTestBase, op_utils.NumpyOpTest):

    def setup(self):
        in_dtype, = self.in_dtypes
        if isinstance(self.beta_range, tuple):
            l, u = self.beta_range
            self.beta = random.uniform(l, u)
        elif self.beta_range is Unspecified:
            self.beta = 1.0
        else:
            self.beta = self.beta_range

        if numpy.dtype(in_dtype).kind != 'f':
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        if in_dtype == 'float16':
            self.check_forward_options.update({'rtol': 2e-3, 'atol': 2e-3})
            self.check_backward_options.update({'rtol': 2e-3, 'atol': 2e-3})
            self.check_double_backward_options.update(
                {'rtol': 1e-2, 'atol': 1e-2})

    def func(self, xp, a):
        in_dtype, = self.in_dtypes
        if xp is numpy:
            ba = self.beta * a
            beta_inv = 1.0 / self.beta
            y = (numpy.fmax(ba, 0) +
                 numpy.log1p(numpy.exp(-numpy.fabs(ba)))) * beta_inv
            return y
        elif self.beta_range is Unspecified:
            return xp.softplus(a)
        else:
            return xp.softplus(a, self.beta)
