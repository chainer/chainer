import chainer
import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import math_utils
from chainerx_tests import op_utils


_logsumexp_params = [
    ((2,), 0),
    ((2,), -1),
    ((2, 3), None),
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 3), -2),
    ((2, 3), (0, 1)),
    ((2, 3), (-2, 1)),
    ((1, 2, 3), None),
    ((1, 2, 3), (1)),
    ((1, 2, 3), (1, 0)),
    ((1, 2, 3), (0, 1, 2)),
]


_invalid_logsumexp_params = [
    # Axis out of bounds
    ((2,), 1),
    ((2,), -2),
    ((2,), (0, 1)),
    ((2, 3), (0, 1, 2)),
    # Duplicate axes
    ((2,), (0, 0)),
    ((2, 3), (0, 0)),
]


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('in_dtypes,out_dtype', [
    (('bool_',), 'int64'),
    (('int8',), 'int64'),
    (('int16',), 'int64'),
    (('int32',), 'int64'),
    (('int64',), 'int64'),
    (('float16',), 'float16'),
    (('float32',), 'float32'),
    (('float64',), 'float64'),

    # TODO(niboshi): Unsigned integer dtypes should result in uint64.
    # Currently chainerx returns int64.
    (('uint8',), 'int64'),
])
@chainer.testing.parameterize_pytest('shape,axis', [
    ((), None),
    ((), ()),
    ((2,), None),
    ((2,), ()),
    ((2,), 0),
    ((2,), (0,)),
    ((2,), (-1,)),
    ((2, 3), None),
    ((2, 3), ()),
    ((2, 3), 0),
    ((2, 3), (0,)),
    ((2, 3), (1,)),
    ((2, 3), (-1,)),
    ((2, 3), (-2,)),
    ((2, 3), (0, 1)),
    ((2, 3), (-2, -1)),
    ((1, 3), None),  # sum over 1-dim axis
    ((0, 3), None),  # sum over 0-dim axis
    # Sum over axes that are in the middle or apart
    ((2, 3, 4), (1,)),
    ((2, 3, 4), (0, 2)),
    # Sum over axes that are apart and/or unsorted
    ((2, 3), (1, 0)),
    ((2, 3, 4), (2, 0)),
    ((2, 3, 4), (2, 0, 1)),
    ((2, 3, 4), (-2, 2, 0)),
])
@chainer.testing.parameterize_pytest('keepdims', [True, False])
@chainer.testing.parameterize_pytest('is_module', [True, False])
class TestSum(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    input = 'random'

    def setup(self):
        super().setup()
        in_dtype, = self.in_dtypes
        if in_dtype == 'float16':
            self.check_forward_options.update({'rtol': 1e-2, 'atol': 1e-2})
            self.check_backward_options.update({'rtol': 1e-2, 'atol': 1e-2})
            self.check_double_backward_options.update(
                {'rtol': 1e-2, 'atol': 1e-2})

    def func(self, xp, a):
        if self.is_module:
            return xp.sum(a, axis=self.axis, keepdims=self.keepdims)
        else:
            return a.sum(axis=self.axis, keepdims=self.keepdims)


@op_utils.op_test(['native:0'])
class TestSumStability(op_utils.NumpyOpTest):

    skip_backward_test = True
    skip_double_backward_test = True

    def generate_inputs(self):
        return numpy.full(2 ** 20, 0.1, dtype=numpy.float32),

    def forward_xp(self, inputs, xp):
        x, = inputs
        if xp is chainerx:
            return x.sum(),
        else:
            return (x[0] * x.size).astype(x.dtype),


@op_utils.op_test(['native:0'])
@chainer.testing.parameterize_pytest('size', list(range(1024)))
class TestSumEachSize(op_utils.NumpyOpTest):

    skip_backward_test = True
    skip_double_backward_test = True

    def generate_inputs(self):
        return numpy.arange(self.size, dtype=numpy.int32) + 1,

    def forward_xp(self, inputs, xp):
        x, = inputs
        return x.sum(),


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(chainerx.DimensionError, ValueError))
@pytest.mark.parametrize('keepdims', [False, True])
@pytest.mark.parametrize('shape,axis', [
    # ((), 0), # TODO(sonots): Fix compatibility
    ((), 1),
    ((), (1,)),
    ((2,), 2),
    ((2,), (2,)),
    ((2,), (-2,)),
    ((2, 3,), (-3,)),
    ((2, 3,), (-3, -4)),
    ((2, 3,), (0, 0)),
    ((2, 3,), (-1, -1)),
    ((2, 3,), (0, 1, 1)),
    ((2, 3,), (0, -2)),
])
def test_sum_invalid(is_module, xp, shape, axis, keepdims, dtype):
    a = array_utils.create_dummy_ndarray(xp, shape, dtype)
    if is_module:
        xp.sum(a, axis=axis, keepdims=keepdims)
    else:
        a.sum(axis=axis, keepdims=keepdims)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shape,axis', _logsumexp_params)
@chainer.testing.parameterize_pytest(
    'in_dtypes,out_dtype', math_utils.in_out_dtypes_math_functions)
class TestSoftmax(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    input = 'random'

    def setup(self):
        super().setup()
        self.check_forward_options.update({'rtol': 3e-3, 'atol': 3e-3})
        self.check_backward_options.update({'rtol': 3e-3, 'atol': 3e-3})

    def forward_xp(self, inputs, xp):
        x, = inputs
        axis = self.axis
        if xp is chainerx:
            return chainerx.softmax(x, axis=axis),
        x = x.astype(self.out_dtype)
        axis = axis if axis is not None else 1
        return numpy.exp(x) / (numpy.exp(x).sum(axis=axis, keepdims=True)),


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest(
    'in_dtypes,out_dtype', math_utils.in_out_dtypes_math_functions)
@chainer.testing.parameterize_pytest('shape,axis', _logsumexp_params)
@chainer.testing.parameterize_pytest('keepdims', [True, False])
class TestLogSumExp(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    input = 'random'

    def setup(self):
        super().setup()
        if self.in_dtypes == 'float16':
            # TODO(imanishi): Support device implementation and remove this.
            self.check_forward_options.update({'rtol': 3e-3, 'atol': 3e-3})

    def forward_xp(self, inputs, xp):
        x, = inputs
        axis = self.axis
        keepdims = self.keepdims
        if xp is chainerx:
            return chainerx.logsumexp(x, axis=axis, keepdims=keepdims),
        x = x.astype(self.out_dtype)
        return numpy.log(numpy.exp(x).sum(axis=axis, keepdims=keepdims)),


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _invalid_logsumexp_params)
@pytest.mark.parametrize('keepdims', [True, False])
# TODO(hvy): Should not overflow for large numbers, add tests
def test_logsumexp_invalid(device, a_shape, axis, keepdims, dtype):
    a = array_utils.create_dummy_ndarray(chainerx, a_shape, dtype)
    with pytest.raises(chainerx.DimensionError):
        chainerx.logsumexp(a, axis=axis, keepdims=keepdims)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shape,axis', _logsumexp_params)
@chainer.testing.parameterize_pytest(
    'in_dtypes,out_dtype', math_utils.in_out_dtypes_math_functions)
class TestLogSoftmax(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    input = 'random'

    def setup(self):
        super().setup()
        self.check_forward_options.update({'rtol': 3e-3, 'atol': 3e-3})
        self.check_backward_options.update({'rtol': 3e-3, 'atol': 3e-3})

    def forward_xp(self, inputs, xp):
        x, = inputs
        axis = self.axis
        if xp is chainerx:
            return chainerx.log_softmax(x, axis=axis),
        x = x.astype(self.out_dtype)
        axis = axis if axis is not None else 1
        return x - numpy.log(numpy.exp(x).sum(axis=axis, keepdims=True)),


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _invalid_logsumexp_params)
def test_log_softmax_invalid(device, a_shape, axis, dtype):
    a = array_utils.create_dummy_ndarray(chainerx, a_shape, dtype)
    with pytest.raises(chainerx.DimensionError):
        return chainerx.log_softmax(a, axis=axis)
