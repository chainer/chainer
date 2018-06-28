import unittest

import chainer
from chainer import np
from chainer import testing


class TestCreationRoutines(unittest.TestCase):

    def test_empty(self):
        a = np.empty((2, 3), dtype=np.float32)
        assert isinstance(a, chainer.Variable)
        assert a.shape == (2, 3)
        assert a.dtype == np.float32

    def test_empty_like(self):
        a = np.empty((2, 3), dtype=np.float32)
        b = np.empty_like(a, dtype=np.float16)
        assert isinstance(b, chainer.Variable)
        assert b.shape == a.shape
        assert b.dtype == np.float16

    @np.testing.numpy_chainer_array_equal()
    def test_eye(self, np):
        return np.eye(5, 4, 2, np.float32)

    @np.testing.numpy_chainer_array_equal()
    def test_full(self, np):
        return np.full((2, 3), 3, np.float32)

    @np.testing.numpy_chainer_array_equal()
    def test_full_like(self, np):
        return np.full_like(np.empty((2, 3), np.float32), 3, np.float16)

    @np.testing.numpy_chainer_array_equal()
    def test_identity(self, np):
        return np.identity(3, np.float64)

    @np.testing.numpy_chainer_array_equal()
    def test_ones(self, np):
        return np.ones((2, 3), np.float32)

    @np.testing.numpy_chainer_array_equal()
    def test_ones_like(self, np):
        return np.ones_like(np.empty((2, 3), np.float32), np.float16)

    @np.testing.numpy_chainer_array_equal()
    def test_zeros(self, np):
        return np.zeros((2, 3), np.float32)

    @np.testing.numpy_chainer_array_equal()
    def test_zeros_like(self, np):
        return np.zeros_like(np.empty((2, 3), np.float32), np.float16)


testing.run_module(__name__, __file__)
