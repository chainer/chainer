import unittest

import cupy
import cupy.sparse
from cupy import testing


class TestCooMatrix(unittest.TestCase):

    def setUp(self):
        data = cupy.array([1, 2, 3, 4], 'f')
        row = cupy.array([0, 0, 1, 2], 'i')
        col = cupy.array([0, 1, 3, 2], 'i')
        self.m = cupy.sparse.coo_matrix((data, (row, col)), shape=(3, 4))

    def test_shape(self):
        self.assertEqual(self.m.shape, (3, 4))

    def test_ndim(self):
        self.assertEqual(self.m.ndim, 2)

    def test_nnz(self):
        self.assertEqual(self.m.nnz, 4)

    def test_str(self):
        self.assertEqual(str(self.m), '''  (0, 0)\t1.0
  (0, 1)\t2.0
  (1, 3)\t3.0
  (2, 2)\t4.0''')
