import unittest

import cupy
import cupy.sparse
from cupy import testing


class TestCscMatrix(unittest.TestCase):

    def setUp(self):
        data = cupy.array([1, 2, 4, 3], 'f')
        indices = cupy.array([0, 0, 2, 1], 'i')
        indptr = cupy.array([0, 1, 2, 3, 4], 'i')
        self.m = cupy.sparse.csc_matrix((data, indices, indptr), shape=(3, 4))

    def test_shape(self):
        self.assertEqual(self.m.shape, (3, 4))

    def test_ndim(self):
        self.assertEqual(self.m.ndim, 2)

    def test_nnz(self):
        self.assertEqual(self.m.nnz, 4)

    def test_str(self):
        self.assertEqual(str(self.m), '''  (0, 0)\t1.0
  (0, 1)\t2.0
  (2, 2)\t4.0
  (1, 3)\t3.0''')

    def test_toarray(self):
        m = self.m.toarray()
        expect = [
            [1, 2, 0, 0],
            [0, 0, 0, 3],
            [0, 0, 4, 0]
        ]
        cupy.testing.assert_allclose(m, expect)


class TestCscMatrixScipyComparison(unittest.TestCase):

    def make(self, xp, sp):
        data = xp.array([1, 2, 4, 3], 'f')
        indices = xp.array([0, 0, 2, 1], 'i')
        indptr = xp.array([0, 1, 2, 3, 4], 'i')
        return sp.csc_matrix((data, indices, indptr), shape=(3, 4))

    def make_unordered(self, xp, sp):
        data = xp.array([1, 2, 3, 4], 'f')
        indices = xp.array([1, 0, 1, 2], 'i')
        indptr = xp.array([0, 0, 0, 2, 4], 'i')
        return sp.csc_matrix((data, indices, indptr), shape=(3, 4))

    @testing.scipy_cupy_allclose(accept_error=False)
    def test_toarray(self, xp, sp):
        m = self.make(xp, sp)
        return m.toarray()

    @testing.scipy_cupy_allclose(accept_error=False)
    def test_tocsc(self, xp, sp):
        m = self.make(xp, sp)
        return m.tocsc().toarray()

    @testing.scipy_cupy_allclose(accept_error=False)
    def test_tocsr(self, xp, sp):
        m = self.make(xp, sp)
        return m.tocsr().toarray()

    @testing.scipy_cupy_allclose(accept_error=False)
    def test_sort_indices(self, xp, sp):
        m = self.make_unordered(xp, sp)
        m.sort_indices()
        return m.indices

    @testing.scipy_cupy_allclose(accept_error=False)
    def test_transpose(self, xp, sp):
        m = self.make(xp, sp)
        return m.transpose().toarray()
