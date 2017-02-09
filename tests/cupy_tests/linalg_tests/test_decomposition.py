import unittest

import numpy

from cupy import testing


@testing.gpu
class TestCholeskyDecomposition(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_L(self, array, xp, dtype):
        a = xp.asarray(array, dtype=dtype)
        return getattr(xp, 'linalg').cholesky(a)

    def test_all(self):
        # A normal positive definite matrix
        self.check_L(numpy.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]]))
        # np.linalg.cholesky only uses a lower triangle of an array
        self.check_L(numpy.array([[1, 2], [1, 9]]))


@testing.gpu
class TestQRDecomposition(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_mode(self, array, xp, dtype, mode, index=None):
        a = xp.asarray(array, dtype=dtype)
        result = getattr(xp, 'linalg').qr(a, mode=mode)
        if type(result) == tuple:
            return result[index]
        else:
            return result

    def test_r_mode(self):
        self.check_mode(numpy.random.randn(2, 3), mode='r')
        self.check_mode(numpy.random.randn(3, 3), mode='r')
        self.check_mode(numpy.random.randn(4, 2), mode='r')

    def test_raw_mode(self):
        self.check_mode(numpy.random.randn(2, 4), mode='raw', index=0)
        self.check_mode(numpy.random.randn(1, 5), mode='raw', index=1)
        self.check_mode(numpy.random.randn(2, 3), mode='raw', index=0)
        self.check_mode(numpy.random.randn(4, 3), mode='raw', index=1)
        self.check_mode(numpy.random.randn(4, 5), mode='raw', index=0)
        self.check_mode(numpy.random.randn(3, 3), mode='raw', index=1)

    def test_complete_mode(self):
        self.check_mode(numpy.random.randn(2, 4), mode='complete', index=0)
        self.check_mode(numpy.random.randn(1, 5), mode='complete', index=1)
        self.check_mode(numpy.random.randn(2, 3), mode='complete', index=0)
        self.check_mode(numpy.random.randn(4, 3), mode='complete', index=1)
        self.check_mode(numpy.random.randn(4, 5), mode='complete', index=0)
        self.check_mode(numpy.random.randn(3, 3), mode='complete', index=1)

    def test_reduced_mode(self):
        self.check_mode(numpy.random.randn(2, 4), mode='reduced', index=0)
        self.check_mode(numpy.random.randn(1, 5), mode='reduced', index=1)
        self.check_mode(numpy.random.randn(2, 3), mode='reduced', index=0)
        self.check_mode(numpy.random.randn(4, 3), mode='reduced', index=1)
        self.check_mode(numpy.random.randn(4, 5), mode='reduced', index=0)
        self.check_mode(numpy.random.randn(3, 3), mode='reduced', index=1)


@testing.gpu
class TestSVD(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_svd(self, array, xp, dtype, full_matrices, index=None):
        a = xp.asarray(array, dtype=dtype)
        result = getattr(xp, 'linalg').svd(a, full_matrices=full_matrices)
        # Use abs in order to support an inverse vector
        if type(result) == tuple:
            return xp.abs(result[index])
        else:
            return xp.abs(result)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_singular(self, array, xp, dtype, full_matrices):
        a = xp.asarray(array, dtype=dtype)
        result = getattr(xp, 'linalg').svd(
            a, full_matrices=full_matrices, compute_uv=False)
        return result

    def check_all(self, a, full_matrices=True):
        self.check_svd(a, full_matrices=full_matrices, index=0)
        self.check_svd(a, full_matrices=full_matrices, index=1)
        self.check_svd(a, full_matrices=full_matrices, index=2)

    def test_svd_full_matrices(self):
        self.check_all(numpy.random.randn(2, 3), full_matrices=True)
        self.check_all(numpy.random.randn(2, 2), full_matrices=True)
        self.check_all(numpy.random.randn(3, 2), full_matrices=True)

    def test_svd_full_matrices(self):
        self.check_all(numpy.random.randn(2, 3), full_matrices=False)
        self.check_all(numpy.random.randn(2, 2), full_matrices=False)
        self.check_all(numpy.random.randn(3, 2), full_matrices=False)

    def test_svd_no_uv(self):
        self.check_singular(numpy.random.randn(2, 3), full_matrices=True)
        self.check_singular(numpy.random.randn(2, 2), full_matrices=True)
        self.check_singular(numpy.random.randn(3, 2), full_matrices=True)

        self.check_singular(numpy.random.randn(2, 3), full_matrices=False)
        self.check_singular(numpy.random.randn(2, 2), full_matrices=False)
        self.check_singular(numpy.random.randn(3, 2), full_matrices=False)
