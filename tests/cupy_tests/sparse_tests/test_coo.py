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

    def test_toarray(self):
        m = self.m.toarray()
        expect = [
            [1, 2, 0, 0],
            [0, 0, 0, 3],
            [0, 0, 4, 0]
        ]
        cupy.testing.assert_allclose(m, expect)


class TestCooMatrixScipyComparison(unittest.TestCase):

    def make(self, xp, sp):
        data = xp.array([1, 2, 3, 4], 'f')
        row = xp.array([0, 0, 1, 2], 'i')
        col = xp.array([0, 1, 3, 2], 'i')
        return sp.coo_matrix((data, (row, col)), shape=(3, 4))

    def make2(self, xp, sp):
        data = xp.array([1, 2, 3, 4], 'f')
        row = xp.array([0, 1, 0, 1], 'i')
        col = xp.array([2, 1, 2, 1], 'i')
        return sp.coo_matrix((data, (row, col)), shape=(3, 4))

    def make3(self, xp, sp):
        data = xp.array([1, 2, 3, 4, 5], 'f')
        row = xp.array([0, 1, 1, 3, 3], 'i')
        col = xp.array([0, 2, 1, 3, 4], 'i')
        return sp.csr_matrix((data, (row, col)), shape=(4, 3))

    @testing.scipy_cupy_allclose(accept_error=False)
    def test_toarray(self, xp, sp):
        m = self.make(xp, sp)
        return m.toarray()

    @testing.scipy_cupy_allclose(accept_error=False)
    def test_tocsc(self, xp, sp):
        m = self.make(xp, sp)
        return m.tocsr().toarray()

    @testing.scipy_cupy_allclose(accept_error=False)
    def test_tocsr(self, xp, sp):
        m = self.make(xp, sp)
        return m.tocsr().toarray()

    @testing.scipy_cupy_allclose(accept_error=False)
    def test_transpose(self, xp, sp):
        m = self.make(xp, sp)
        return m.transpose().toarray()
