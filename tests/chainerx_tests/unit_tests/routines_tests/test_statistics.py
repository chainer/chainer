import chainer
import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import math_utils
from chainerx_tests import op_utils


def test_max_amax():
    assert chainerx.amax is chainerx.max


_minmax_params = [
    # --- single axis
    # input, axis
    (numpy.asarray(0), None),
    (numpy.asarray(-1), None),
    (numpy.asarray(float('inf')), None),
    (numpy.asarray(float('nan')), None),
    (numpy.asarray(-float('inf')), None),
    (numpy.asarray([4, 1, 4, 1]), None),
    (numpy.asarray([4, 1, 4, 1]), 0),
    (numpy.asarray([[4, 4, 1, 1], [4, 1, 4, 1]]), 0),
    (numpy.asarray([[4, 4, 1, 1], [4, 1, 4, 1]]).T, 1),
    (numpy.asarray([-0.0, +0.0, +0.0, -0.0]), None),
    (numpy.asarray([[True, True, False, False],
                    [True, False, True, False]]), 0),
    (numpy.ones((2, 3)), 1),
    (numpy.ones((2, 3)), -2),
    # --- multiple axes
    # input, axis
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (0, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (-2, -1)),
]


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape,axis': [
            ((), None),
            ((4,), None),
            ((4,), 0),
            ((4, 2), None),
            ((4, 2), 0),
            ((4, 2), 1),
            ((4, 2), -2),
            ((4, 3), (0, 1)),
            ((4, 3), (-2, -1)),
        ],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                1, chainerx.testing.all_dtypes)),
        'is_module': [True, False],
    }) +
    chainer.testing.product({
        'array,axis': _minmax_params,
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                1, chainerx.testing.all_dtypes)),
        'is_module': [True, False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestMax(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    dodge_nondifferentiable = True

    def generate_inputs(self):
        in_dtype, = self.in_dtypes
        if hasattr(self, 'array'):
            return self.array.astype(in_dtype),
        return array_utils.uniform(self.shape, in_dtype),

    def func(self, xp, a):
        if self.is_module:
            return xp.max(a, self.axis)
        else:
            return a.max(self.axis)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('array,axis', [
    (numpy.ones((2, 3)), 2),
    (numpy.ones((2, 3)), -3),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (-3, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 2)),
])
@pytest.mark.parametrize('dtype', chainerx.testing.all_dtypes)
@pytest.mark.parametrize('is_module', [True, False])
def test_max_invalid_shapes_and_axis(device, array, axis, dtype, is_module):
    a = chainerx.array(array).astype(dtype)
    with pytest.raises(chainerx.DimensionError):
        if is_module:
            chainerx.max(a, axis)
        else:
            a.max(axis)


def test_min_amin():
    assert chainerx.amin is chainerx.min


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape,axis': [
            ((), None),
            ((4,), None),
            ((4,), 0),
            ((4, 2), None),
            ((4, 2), 0),
            ((4, 2), 1),
            ((4, 2), -2),
            ((4, 3), (0, 1)),
            ((4, 3), (-2, -1)),
        ],
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                1, chainerx.testing.all_dtypes)),
        'is_module': [True, False],
    }) +
    chainer.testing.product({
        'array,axis': _minmax_params,
        'in_dtypes,out_dtype': (
            dtype_utils.make_same_in_out_dtypes(
                1, chainerx.testing.all_dtypes)),
        'is_module': [True, False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestMin(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    dodge_nondifferentiable = True

    def generate_inputs(self):
        in_dtype, = self.in_dtypes
        if hasattr(self, 'array'):
            return self.array.astype(in_dtype),
        return array_utils.uniform(self.shape, in_dtype),

    def func(self, xp, a):
        if self.is_module:
            return xp.min(a, self.axis)
        else:
            return a.min(self.axis)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('array,axis', [
    (numpy.ones((2, 3)), 2),
    (numpy.ones((2, 3)), -3),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (-3, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 2)),
])
@pytest.mark.parametrize('dtype', chainerx.testing.all_dtypes)
@pytest.mark.parametrize('is_module', [True, False])
def test_min_invalid_shapes_and_axis(device, array, axis, dtype, is_module):
    a = chainerx.array(array).astype(dtype)
    with pytest.raises(chainerx.DimensionError):
        if is_module:
            chainerx.min(a, axis)
        else:
            a.min(axis)


_mean_var_params = \
    chainer.testing.product({
        'shape,axis': [
            ((), None),
            (1, 0),
            ((2, 1, 3), (1, 2)),
            ((1, 1, 1), (0, 1, 2)),
            ((2, 3), None),
            ((1, 2, 3), (0, 2)),
            ((2, 2, 2, 2), (2, 1, 0)),
            ((1, 1, 1), (-1))],
        'in_dtypes,out_dtype': math_utils.in_out_dtypes_math_functions,
        'input': ['random'],
        'contiguous': [None, 'C'],
    }) + chainer.testing.product({
        'shape,axis': [((2, 3), None)],
        'in_dtypes,out_dtype': math_utils.in_out_float_dtypes_math_functions,
        'input': [1.57, 2, 3.14, float('inf'), -float('inf'), float('nan')],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _mean_var_params
))
class TestMean(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.mean(a, self.axis)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    _mean_var_params
))
class TestVar(math_utils.UnaryMathTestBase, op_utils.NumpyOpTest):

    def func(self, xp, a):
        return xp.var(a, self.axis)


def apply_func(is_module, func, xp, device, input, axis, dtypes):
    (in_dtype,), out_dtype = dtypes
    try:
        a_np = input.astype(in_dtype)
    except (ValueError, OverflowError):
        return xp.zeros(())  # invalid combination of data and dtype

    a = xp.array(a_np)
    a = func(is_module, xp, a, axis)
    if xp is numpy:
        a = dtype_utils.cast_if_numpy_array(xp, a, out_dtype)
    return a


def compute_mean(is_module, xp, a, axis):
    return xp.mean(a, axis) if is_module else a.mean(axis)


def compute_var(is_module, xp, a, axis):
    return xp.var(a, axis) if is_module else a.var(axis)


@chainerx.testing.numpy_chainerx_array_equal(strides_check=False)
@pytest.mark.parametrize('input,axis', [
    # --- single axis
    # input, axis
    # valid params
    (numpy.asarray(0), None),
    (numpy.asarray(-1), None),
    (numpy.asarray(float('inf')), None),
    (numpy.asarray(float('nan')), None),
    (numpy.asarray(-float('inf')), None),
    (numpy.asarray([4, 1, 4, 1]), None),
    (numpy.asarray([4, 1, 4, 1]), 0),
    (numpy.asarray([[4, 4, 1, 1], [4, 1, 4, 1]]), 0),
    (numpy.asarray([[4, 4, 1, 1], [4, 1, 4, 1]]).T, 1),
    (numpy.asarray([-0.0, +0.0, +0.0, -0.0]), None),
    (numpy.asarray([[True, True, False, False],
                    [True, False, True, False]]), 0),
    (numpy.ones((2, 0, 3)), 2),
    (numpy.ones((2, 3)), 1),
    (numpy.ones((2, 3)), -2),
    # --- multiple axes
    # input, axis
    # valid params
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (0, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (-2, -1)),
])
@pytest.mark.parametrize('dtypes', math_utils.in_out_dtypes_math_functions)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('func', [
    compute_mean,
    compute_var,
])
# TODO(kshitij12345): Remove strides_check=False
def test_valid_stats(is_module, func, xp, device, input, axis, dtypes):
    return apply_func(is_module, func, xp, device, input, axis, dtypes)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(IndexError, ValueError, chainerx.DimensionError),
    strides_check=False)
@pytest.mark.parametrize('input,axis', [
    # --- single axis
    # input, axis
    # invalid params
    (numpy.ones((0,)), None),
    (numpy.ones((2, 0, 3)), 1),
    (numpy.ones((2, 0, 3)), None),
    (numpy.ones((2, 3)), 2),
    (numpy.ones((2, 3)), -3),
    # --- multiple axes
    # input, axis
    # invalid params
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (-3, 1)),
    (numpy.asarray([[1, 4, 3, 1], [4, 6, 3, 2], [2, 3, 6, 1]]), (1, 2)),
])
@pytest.mark.parametrize('dtypes', math_utils.in_out_dtypes_math_functions)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('func', [
    compute_mean,
    compute_var,
])
# TODO(kshitij12345): Remove strides_check=False
def test_invalid_stats(is_module, func, xp, device, input, axis, dtypes):
    return apply_func(is_module, func, xp, device, input, axis, dtypes)
