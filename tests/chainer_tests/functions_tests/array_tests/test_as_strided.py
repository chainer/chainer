import unittest

from chainer import testing, Variable, grad

import numpy as np
import cupy as cp

from chainer.functions import as_strided
from chainer.functions.array.as_strided import _stride_array


@testing.parameterize(
    {'dtype': np.float16},
    {'dtype': np.float32},
    {'dtype': np.float64},
    {'dtype': np.int16},
    {'dtype': np.int32},
    {'dtype': np.int64}
)
class TestStrideArray(unittest.TestCase):
    def test_flip_cpu(self):
        x = np.arange(4, dtype=self.dtype)
        y = _stride_array(x, (4,), (-1,), 3)  # [3, 2, 1, 0]
        y_expected = np.flip(x, axis=0)
        assert (y == y_expected).all()

    @testing.attr.gpu
    def test_flip_gpu(self):
        x = cp.arange(4, dtype=self.dtype)
        y = _stride_array(x, (4,), (-1,), 3)
        y_expected = cp.flip(x, axis=0)
        assert (y == y_expected).all()

    def test_broadcast_cpu(self):
        x = np.arange(12, dtype=self.dtype).reshape((3, 4)).copy()  # [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        y = _stride_array(x, (2, 3, 4), (0, 4, 1), 0)
        y_expected = np.broadcast_to(x, (2, 3, 4))
        assert (y == y_expected).all()

    @testing.attr.gpu
    def test_broadcast_gpu(self):
        x = cp.arange(12, dtype=self.dtype).reshape((3, 4)).copy()
        y = _stride_array(x, (2, 3, 4), (0, 4, 1), 0)
        y_expected = cp.broadcast_to(x, (2, 3, 4))
        assert (y == y_expected).all()

    def test_unstride_cpu(self):
        x = np.flip(np.arange(12, dtype=self.dtype).reshape((3, 4)), 0)
        y = _stride_array(x, (12,), (1,), 0)
        y_expected = np.arange(12, dtype=self.dtype)
        assert (y == y_expected).all()

    @testing.attr.gpu
    def test_unstride_gpu(self):
        x = cp.flip(cp.arange(12, dtype=self.dtype).reshape((3, 4)), 0)
        y = _stride_array(x, (12,), (1,), 0)
        y_expected = cp.arange(12, dtype=self.dtype)
        assert (y == y_expected).all()

    def test_general_stride_cpu(self):
        x = np.arange(8, dtype=self.dtype)
        y = _stride_array(x, (3, 3), (-1, 2), 3)
        y_expected = np.array(
            [[3, 5, 7],
             [2, 4, 6],
             [1, 3, 5]],
            dtype=self.dtype
        )
        assert (y == y_expected).all()

    @testing.attr.gpu
    def test_general_stride_gpu(self):
        x = cp.arange(8, dtype=self.dtype)
        y = _stride_array(x, (3, 3), (-1, 2), 3)
        y_expected = cp.array(
            [[3, 5, 7],
             [2, 4, 6],
             [1, 3, 5]],
            dtype=self.dtype
        )
        assert (y == y_expected).all()

    def test_invalid_negative_index_cpu(self):
        x = np.arange(8, dtype=self.dtype)
        with self.assertRaises(ValueError):
            _stride_array(x, (3, 3), (-1, 2), 1)

    @testing.attr.gpu
    def test_invalid_negative_index_gpu(self):
        x = cp.arange(8, dtype=self.dtype)
        with self.assertRaises(ValueError):
            _stride_array(x, (3, 3), (-1, 2), 1)


@testing.parameterize(
    {'dtype': np.float16},
    {'dtype': np.float32},
    {'dtype': np.float64},
    {'dtype': np.int16},
    {'dtype': np.int32},
    {'dtype': np.int64}
)
class TestAsStridedForward(unittest.TestCase):
    def test_flip_forward_cpu(self):
        x = np.arange(4, dtype=self.dtype)
        v = Variable(x)
        y = as_strided(v, (4,), (-1,), 3)
        y_expected = np.flip(x, axis=0)
        assert (y.array == y_expected).all()

    @testing.attr.gpu
    def test_flip_forward_gpu(self):
        x = cp.arange(4, dtype=self.dtype)
        v = Variable(x)
        y = as_strided(v, (4,), (-1,), 3)
        y_expected = cp.flip(x, axis=0)
        assert (y.array == y_expected).all()

    def test_broadcast_forward_cpu(self):
        x = np.arange(12, dtype=self.dtype).reshape((3, 4)).copy()
        v = Variable(x)
        y = as_strided(v, (2, 3, 4), (0, 4, 1), 0)
        y_expected = np.broadcast_to(x, (2, 3, 4))
        assert (y.array == y_expected).all()

    @testing.attr.gpu
    def test_broadcast_forward_gpu(self):
        x = cp.arange(12, dtype=self.dtype).reshape((3, 4)).copy()
        v = Variable(x)
        y = as_strided(v, (2, 3, 4), (0, 4, 1), 0)
        y_expected = cp.broadcast_to(x, (2, 3, 4))
        assert (y.array == y_expected).all()

    def test_unstride_forward_cpu(self):
        x = np.flip(np.arange(12, dtype=self.dtype).reshape((3, 4)), 0)
        v = Variable(x)
        y = as_strided(v, (12,), (1,), 0)
        y_expected = np.arange(12, dtype=self.dtype)
        assert (y.array == y_expected).all()

    @testing.attr.gpu
    def test_unstride_forward_gpu(self):
        x = cp.flip(cp.arange(12, dtype=self.dtype).reshape((3, 4)), 0)
        v = Variable(x)
        y = as_strided(v, (12,), (1,), 0)
        y_expected = cp.arange(12, dtype=self.dtype)
        assert (y.array == y_expected).all()

    def test_general_stride_forward_cpu(self):
        x = _stride_array(np.arange(8, dtype=self.dtype), (3, 3), (-1, 2), 3)
        # [[3., 5., 7.], [2., 4., 6.], [1., 3., 5.]]
        v = Variable(x)
        y = as_strided(v, (3, 3), (1, 2), 0)
        # [[0., 2., 4.], [1., 3., 5.,], [2., 4., 6.]]
        y_expected = _stride_array(np.arange(8, dtype=self.dtype), (3, 3), (1, 2), 0)
        assert (y.array == y_expected).all()

    @testing.attr.gpu
    def test_general_stride_forward_gpu(self):
        x = _stride_array(cp.arange(8, dtype=self.dtype), (3, 3), (-1, 2), 3)
        # [[3., 5., 7.], [2., 4., 6.], [1., 3., 5.]]
        v = Variable(x)
        y = as_strided(v, (3, 3), (1, 2), 0)
        # [[0., 2., 4.], [1., 3., 5.,], [2., 4., 6.]]
        y_expected = _stride_array(cp.arange(8, dtype=self.dtype), (3, 3), (1, 2), 0)
        assert (y.array == y_expected).all()


@testing.parameterize(
    {'dtype': np.float16},
    {'dtype': np.float32},
    {'dtype': np.float64}
)
class TestAsStridedBackward(unittest.TestCase):
    def test_flip_backward_cpu(self):
        x = np.arange(4, dtype=self.dtype)
        v = Variable(x)
        y = as_strided(v, (4,), (-1,), 3)
        y.grad = np.ones((4,), dtype=self.dtype)
        gx, = grad((y,), (v,))
        assert (gx.array == np.ones((4,), dtype=self.dtype)).all()

    @testing.attr.gpu
    def test_flip_backward_gpu(self):
        x = cp.arange(4, dtype=self.dtype)
        v = Variable(x)
        y = as_strided(v, (4,), (-1,), 3)
        y.grad = cp.ones((4,), dtype=self.dtype)
        gx, = grad((y,), (v,))
        assert (gx.array == cp.ones((4,), dtype=self.dtype)).all()

    def test_broadcast_backward_cpu(self):
        x = np.arange(12, dtype=self.dtype).reshape((3, 4)).copy()
        v = Variable(x)
        y = as_strided(v, (2, 3, 4), (0, 4, 1), 0)
        y.grad = np.ones((2, 3, 4), dtype=self.dtype)
        gx, = grad((y,), (v,))
        assert (gx.array == np.ones(x.shape, dtype=self.dtype) * 2).all()

    @testing.attr.gpu
    def test_broadcast_backward_gpu(self):
        x = cp.arange(12, dtype=self.dtype).reshape((3, 4)).copy()
        v = Variable(x)
        y = as_strided(v, (2, 3, 4), (0, 4, 1), 0)
        y.grad = cp.ones((2, 3, 4), dtype=self.dtype)
        gx, = grad((y,), (v,))
        assert (gx.array == cp.ones(x.shape, dtype=self.dtype) * 2).all()

    def test_unstride_backward_cpu(self):
        x = np.flip(np.arange(12, dtype=self.dtype).reshape((3, 4)), 0)
        v = Variable(x)
        y = as_strided(v, (12,), (1,), 0)
        y.grad = np.ones((12,), dtype=self.dtype)
        gx, = grad((y,), (v,))
        assert (gx.array == np.ones(x.shape, dtype=self.dtype)).all()

    @testing.attr.gpu
    def test_unstride_backward_gpu(self):
        x = cp.flip(cp.arange(12, dtype=self.dtype).reshape((3, 4)), 0)
        v = Variable(x)
        y = as_strided(v, (12,), (1,), 0)
        y.grad = cp.ones((12,), dtype=self.dtype)
        gx, = grad((y,), (v,))
        assert (gx.array == cp.ones(x.shape, dtype=self.dtype)).all()

    def test_general_stride_backward_cpu(self):
        x = _stride_array(np.arange(8, dtype=self.dtype), (3, 3), (-1, 2), 3)
        # [[3., 5., 7.], [2., 4., 6.], [1., 3., 5.]]
        v = Variable(x)
        y = as_strided(v, (3, 3), (1, 2), 0)
        # [[0., 2., 4.], [1., 3., 5.,], [2., 4., 6.]]
        y.grad = np.ones(y.shape, dtype=self.dtype)
        gx, = grad((y,), (v,))
        assert (gx.array == np.array([[0.5, 0.5, 0.], [2., 2., 1.], [1., 0.5, 0.5]])).all()

    @testing.attr.gpu
    def test_general_stride_backward_gpu(self):
        x = _stride_array(cp.arange(8, dtype=self.dtype), (3, 3), (-1, 2), 3)
        # [[3., 5., 7.], [2., 4., 6.], [1., 3., 5.]]
        v = Variable(x)
        y = as_strided(v, (3, 3), (1, 2), 0)
        # [[0., 2., 4.], [1., 3., 5.,], [2., 4., 6.]]
        y.grad = cp.ones(y.shape, dtype=self.dtype)
        gx, = grad((y,), (v,))
        assert (gx.array == cp.array([[0.5, 0.5, 0.], [2., 2., 1.], [1., 0.5, 0.5]])).all()


@testing.parameterize(
    {'dtype': np.int16},
    {'dtype': np.int32},
    {'dtype': np.int64}
)
class TestAsStridedBackwardInvalidType(unittest.TestCase):
    def test_flip_backward_cpu(self):
        x = np.arange(4, dtype=self.dtype)
        v = Variable(x)
        y = as_strided(v, (4,), (-1,), 3)
        y.grad = np.ones((4,), dtype=self.dtype)
        with self.assertRaises(TypeError):
            gx, = grad((y,), (v,))

    @testing.attr.gpu
    def test_flip_backward_gpu(self):
        x = cp.arange(4, dtype=self.dtype)
        v = Variable(x)
        y = as_strided(v, (4,), (-1,), 3)
        y.grad = cp.ones((4,), dtype=self.dtype)
        with self.assertRaises(TypeError):
            gx, = grad((y,), (v,))

    def test_broadcast_backward_cpu(self):
        x = np.arange(12, dtype=self.dtype).reshape((3, 4)).copy()
        v = Variable(x)
        y = as_strided(v, (2, 3, 4), (0, 4, 1), 0)
        y.grad = np.ones((2, 3, 4), dtype=self.dtype)
        with self.assertRaises(TypeError):
            gx, = grad((y,), (v,))

    @testing.attr.gpu
    def test_broadcast_backward_gpu(self):
        x = cp.arange(12, dtype=self.dtype).reshape((3, 4)).copy()
        v = Variable(x)
        y = as_strided(v, (2, 3, 4), (0, 4, 1), 0)
        y.grad = cp.ones((2, 3, 4), dtype=self.dtype)
        with self.assertRaises(TypeError):
            gx, = grad((y,), (v,))

    def test_unstride_backward_cpu(self):
        x = np.flip(np.arange(12, dtype=self.dtype).reshape((3, 4)), 0)
        v = Variable(x)
        y = as_strided(v, (12,), (1,), 0)
        y.grad = np.ones((12,), dtype=self.dtype)
        with self.assertRaises(TypeError):
            gx, = grad((y,), (v,))

    @testing.attr.gpu
    def test_unstride_backward_gpu(self):
        x = cp.flip(cp.arange(12, dtype=self.dtype).reshape((3, 4)), 0)
        v = Variable(x)
        y = as_strided(v, (12,), (1,), 0)
        y.grad = cp.ones((12,), dtype=self.dtype)
        with self.assertRaises(TypeError):
            gx, = grad((y,), (v,))

    def test_general_stride_backward_cpu(self):
        x = _stride_array(np.arange(8, dtype=self.dtype), (3, 3), (-1, 2), 3)
        # [[3., 5., 7.], [2., 4., 6.], [1., 3., 5.]]
        v = Variable(x)
        y = as_strided(v, (3, 3), (1, 2), 0)
        # [[0., 2., 4.], [1., 3., 5.,], [2., 4., 6.]]
        y.grad = np.ones(y.shape, dtype=self.dtype)
        with self.assertRaises(TypeError):
            gx, = grad((y,), (v,))

    @testing.attr.gpu
    def test_general_stride_backward_gpu(self):
        x = _stride_array(cp.arange(8, dtype=self.dtype), (3, 3), (-1, 2), 3)
        # [[3., 5., 7.], [2., 4., 6.], [1., 3., 5.]]
        v = Variable(x)
        y = as_strided(v, (3, 3), (1, 2), 0)
        # [[0., 2., 4.], [1., 3., 5.,], [2., 4., 6.]]
        y.grad = cp.ones(y.shape, dtype=self.dtype)
        with self.assertRaises(TypeError):
            gx, = grad((y,), (v,))


testing.run_module(__name__, __file__)
