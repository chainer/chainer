import unittest

import numpy

from cupy import testing


@testing.gpu
class TestSVD(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_sigma(self, array, xp, dtype):
        a = xp.asarray(array, dtype=dtype)
        _, s, _ = getattr(xp, 'linalg').svd(a)
        return s

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_U(self, array, xp, dtype):
        a = xp.asarray(array, dtype=dtype)
        u, _, _ = getattr(xp, 'linalg').svd(a)
        return xp.abs(u)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_VT(self, array, xp, dtype):
        a = xp.asarray(array, dtype=dtype)
        u, _, _ = getattr(xp, 'linalg').svd(a)
        return xp.abs(u)

    def check_all(self, a):
        self.check_sigma(a)
        self.check_U(a)
        self.check_VT(a)

    def test_svd(self):
        self.check_all(numpy.random.randn(2, 3))
        self.check_all(numpy.random.randn(2, 2))
        self.check_all(numpy.random.randn(3, 2))
