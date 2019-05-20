import chainer

import chainerx
import chainerx.testing

from chainerx_tests import dtype_utils
from chainerx_tests import math_utils
from chainerx_tests import op_utils


_trigonometric_hyperbolic_params = (
    chainer.testing.product({
        'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
        'input': [-2, 0, 2],
        'contiguous': [None, 'C'],
    }) + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
        'input': [1.57, 2, 3.14, float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
)


def _make_inverse_trig_params(name):
    # Makes test parameters for inverse trigonometric functions

    inverse_trig_differentiable_inputs = {
        'arcsin': [-0.9, 0, 0.9],
        'arccos': [-0.9, 0, 0.9],
        'arctan': [-3, -0.2, 0, 0.2, 3],
        'arcsinh': [-3, -0.2, 0, 0.2, 3],
        'arccosh': [1.2, 3],
        'arctanh': [-0.9, 0, 0.9],
    }

    inverse_trig_nondifferentiable_inputs = {
        'arcsin': [-3, -1, 1, 3],
        'arccos': [-3, -1, 1, 3],
        'arctan': [],
        'arcsinh': [],
        'arccosh': [-3, 0, 0.2, 1],
        'arctanh': [-3, -1, 1, 3],
    }

    nonfinite_numbers = [float('inf'), -float('inf'), float('nan')]

    return (
        # Various shapes and differentiable inputs
        chainer.testing.product({
            'shape': [(), (0,), (1,), (2, 0, 3), (1, 1, 1), (2, 3)],
            'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
            'input': inverse_trig_differentiable_inputs[name],
            'contiguous': [None, 'C'],
        })
        +
        # Nondifferentiable inputs
        chainer.testing.product({
            'shape': [(2, 3)],
            'in_dtypes,out_dtype': (
                math_utils.in_out_float_dtypes_math_functions),
            'input': (
                inverse_trig_nondifferentiable_inputs[name]
                + nonfinite_numbers),
            'skip_backward_test': [True],
            'skip_double_backward_test': [True],
        }))


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _trigonometric_hyperbolic_params
))
class TestSin(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.sin(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _trigonometric_hyperbolic_params
))
class TestCos(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.cos(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _trigonometric_hyperbolic_params
))
class TestTan(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    dodge_nondifferentiable = True
    check_backward_options = {'atol': 3e-5}

    def func(self, xp, a):
        return xp.tan(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _make_inverse_trig_params('arcsin')
))
class TestArcsin(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.arcsin(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _make_inverse_trig_params('arccos')
))
class TestArccos(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.arccos(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _make_inverse_trig_params('arctan')
))
class TestArctan(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.arctan(a)


# Since the gradient of arctan2 is quite flaky.
# for smaller values especially `float16`.
@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'in_shapes': math_utils.shapes_combination_binary,
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': [1],
        'input_rhs': [2],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
    # Differentiable points
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': [-3, -0.75, 0.75, 3],
        'input_rhs': [-3, -0.75, 0.75, 3],
    })
    # Mixed dtypes
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_binary_functions,
        'input_lhs': [-1.],
        'input_rhs': [-1.],
    })
    # Special values
    + chainer.testing.product({
        'in_shapes': [((2, 3), (2, 3))],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                2, chainerx.testing.float_dtypes)),
        'input_lhs': ['random', float('inf'), -float('inf'), float('nan'),
                      +0.0, -0.0],
        'input_rhs': ['random', float('inf'), -float('inf'), float('nan'),
                      +0.0, -0.0],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestArctan2(math_utils.BinaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a, b):
        return xp.arctan2(a, b)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _trigonometric_hyperbolic_params
))
class TestSinh(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.sinh(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _trigonometric_hyperbolic_params
))
class TestCosh(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.cosh(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _trigonometric_hyperbolic_params
))
class TestTanh(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.tanh(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _make_inverse_trig_params('arcsinh')
))
class TestArcsinh(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.arcsinh(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _make_inverse_trig_params('arccosh')
))
class TestArccosh(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.arccosh(a)
