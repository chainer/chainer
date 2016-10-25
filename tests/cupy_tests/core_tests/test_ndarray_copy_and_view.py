import unittest

import numpy

from cupy import testing


@testing.gpu
class TestArrayCopyAndView(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_array_equal()
    def test_view(self, xp):
        a = testing.shaped_arange((4,), xp, dtype=numpy.float32)
        b = a.view(dtype=numpy.int32)
        b[:] = 0
        return a

    @testing.numpy_cupy_array_equal()
    def test_flatten(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.flatten()

    @testing.numpy_cupy_array_equal()
    def test_flatten_copied(self, xp):
        a = testing.shaped_arange((4,), xp)
        b = a.flatten()
        a[:] = 1
        return b

    @testing.numpy_cupy_array_equal()
    def test_transposed_flatten(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp).transpose(2, 0, 1)
        return a.flatten()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_fill(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        a.fill(1)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_transposed_fill(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = a.transpose(2, 0, 1)
        b.fill(1)
        return b

    @testing.for_all_dtypes(name='src_dtype', no_complex=True)
    @testing.for_all_dtypes(name='dst_dtype')
    @testing.numpy_cupy_array_equal()
    def test_astype(self, xp, src_dtype, dst_dtype):
        a = testing.shaped_arange((2, 3, 4), xp, src_dtype)
        return a.astype(dst_dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal1(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return a.diagonal(1, 2, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal2(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return a.diagonal(-1, 2, 0)
