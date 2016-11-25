import unittest

import numpy

from cupy import get_array_module
from cupy import testing


def wrap_scatter(a, ind, v, axis=None, mode=''):
    if get_array_module(a) is numpy:
        if mode == 'update':
            axis %= a.ndim
            ind = [slice(None)] * axis +\
                [ind] + [slice(None)] * (a.ndim - axis - 1)
            a[ind] = v
    else:
        if mode == 'update':
            a.scatter_update(ind, v, axis)


def compute_v_shape(in_shape, indices_shape, axis):
    if len(in_shape) == 0:
        return indices_shape
    else:
        axis %= len(in_shape)
        lshape = in_shape[:axis]
        rshape = in_shape[axis + 1:]
    return lshape + indices_shape + rshape


@testing.parameterize(
    *testing.product({
        'indices': [2, [0, 1], -1, [-1, -2], [[1, 0], [2, 3]]],
        'axis': [0, 1, 2, -1, -2],
    })
)
@testing.gpu
class TestScatterUpdate(unittest.TestCase):

    shape = (4, 4, 5)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_scatter_update(self, xp, dtype):
        a = xp.zeros(self.shape, dtype=dtype)
        v_shape = compute_v_shape(
            self.shape, numpy.array(self.indices).shape, self.axis)
        v = testing.shaped_arange(v_shape, xp, dtype)
        wrap_scatter(a, self.indices, v, self.axis, mode='update')
        return a


@testing.parameterize(
    {'indices_shape': (2,), 'axis': 0, 'v_shape': (1,)},
    {'indices_shape': (2, 2), 'axis': 1, 'v_shape': (5,)},
)
@testing.gpu
class TestScatterUpdateParamterized(unittest.TestCase):

    shape = (3, 4, 5)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_scatter_update(self, xp, dtype):
        a = xp.zeros(self.shape, dtype=dtype)
        m = a.shape[self.axis]
        indices = testing.shaped_arange(
            self.indices_shape, xp, numpy.int32) % m
        v = testing.shaped_arange(self.v_shape, xp, dtype=dtype)
        wrap_scatter(a, indices, v, self.axis, mode='update')
        return a


@testing.parameterize(
    *testing.product({
        'shape': [(3, 4, 5)],
        'indices': [(2,)],
        'axis': [1],
        'v_shape': [(2, 3), (3,)],
        'mode': ['update'],
    })
)
@testing.gpu
class TestScatterOpErrorMismatch(unittest.TestCase):

    @testing.numpy_cupy_raises()
    def test_shape_mismatch(self, xp):
        a = testing.shaped_arange(self.shape, xp, numpy.float32)
        i = testing.shaped_arange(self.indices, xp, numpy.int32) % 3
        v = testing.shaped_arange(self.v_shape, xp, numpy.float32)
        wrap_scatter(a, i, v, self.axis)
