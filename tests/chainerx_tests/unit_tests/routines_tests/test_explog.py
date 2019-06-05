import chainer

from chainerx_tests import math_utils
from chainerx_tests import op_utils


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (1,), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
        'input': [0, 2, -2],
    })
    # Special shapes (array.size = 0)
    + chainer.testing.product({
        'shape': [(0), (2, 0, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
        'input': [0, 2, -2],
        'check_numpy_strides_compliance': [False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
        'input': [float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestExp(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.exp(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (1,), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
        'input': [0, 2, -2],
    })
    # Special shapes (array.size = 0)
    + chainer.testing.product({
        'shape': [(0), (2, 0, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
        'input': [0, 2, -2],
        'check_numpy_strides_compliance': [False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
        'input': [float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestExpm1(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.expm1(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (1,), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
        'input': [0, 2, -2],
    })
    # Special shapes (array.size = 0)
    + chainer.testing.product({
        'shape': [(0), (2, 0, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
        'input': [0, 2, -2],
        'check_numpy_strides_compliance': [False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
        'input': [float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestExp2(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.exp2(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (1,), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
        'input': [1, 3],
    })
    # Special shapes (array.size = 0)
    + chainer.testing.product({
        'shape': [(0,), (2, 0, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
        'input': [1, 3],
        'check_numpy_strides_compliance': [False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
        'input': [float('inf'), -float('inf'), float('nan'), -1, 0],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestLog(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.log(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (1,), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
        'input': [1, 3],
    })
    # Special shapes (array.size = 0)
    + chainer.testing.product({
        'shape': [(0,), (2, 0, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
        'input': [1, 3],
        'check_numpy_strides_compliance': [False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
        'input': [float('inf'), -float('inf'), float('nan'), -1, 0],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestLog10(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.log10(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (1,), (1, 1, 1), (2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
        'input': [1, 3],
    })
    # Special shapes (array.size = 0)
    + chainer.testing.product({
        'shape': [(0,), (2, 0, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
        'input': [1, 3],
        'check_numpy_strides_compliance': [False],
    })
    # Special values
    + chainer.testing.product({
        'shape': [(2, 3)],
        'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
        'input': [float('inf'), -float('inf'), float('nan'), -1, 0],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestLog1p(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.log1p(a)
