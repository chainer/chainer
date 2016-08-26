import unittest

import numpy
import scipy.sparse

import cupy
import cupy.sparse


class TestCsrMatrix(unittest.TestCase):

    def setUp(self):
        data = cupy.array([1, 2, 3, 4], 'f')
        indices = cupy.array([0, 1, 3, 2], 'i')
        indptr = cupy.array([0, 2, 3, 4], 'i')
        self.m = cupy.sparse.csr_matrix((data, indices, indptr), shape=(3, 4))

    def test_nnz(self):
        self.assertEqual(self.m.nnz, 4)

    def test_str(self):
        self.assertEqual(str(self.m), '''  (0, 0)\t1.0
  (0, 1)\t2.0
  (1, 3)\t3.0
  (2, 2)\t4.0''')
