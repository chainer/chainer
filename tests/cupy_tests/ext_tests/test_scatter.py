import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestScatter(unittest.TestCase):

    def test_scatter_update(self):
        a = cupy.zeros((3,))
        i = cupy.array([1, 2], numpy.int32)
        v = cupy.array([2., 1.])
        cupy.scatter_update(a, i, v)
        testing.assert_array_equal(a, cupy.array([0, 2, 1]))
