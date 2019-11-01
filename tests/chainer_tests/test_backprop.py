import unittest

import mock
import numpy as np
import pytest

import chainer
from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr
import chainerx


class TestBackward(unittest.TestCase):

    def test_no_output(self):
        chainer.backward([])
        chainer.backward([], [])

    def check_multiple_output_1arg(self, xp, skip_retain_grad_test=False):
        x = chainer.Variable(xp.array([1, 2], np.float32))
        h = x * 2
        y0 = h * 3
        y1 = h * 4
        y0.grad = xp.array([1, 10], np.float32)
        y1.grad = xp.array([100, 1000], np.float32)
        chainer.backward([y0, y1])
        testing.assert_allclose(x.grad, np.array([806, 8060], np.float32))
        if skip_retain_grad_test:
            return
        assert y0.grad is None
        assert y1.grad is None

    def check_multiple_output_2args(self, xp, skip_retain_grad_test=False):
        x = chainer.Variable(xp.array([1, 2], np.float32))
        h = x * 2
        y0 = h * 3
        y1 = h * 4
        gy0 = chainer.Variable(xp.array([1, 10], np.float32))
        gy1 = chainer.Variable(xp.array([100, 1000], np.float32))
        chainer.backward([y0, y1], [gy0, gy1])
        testing.assert_allclose(x.grad, np.array([806, 8060], np.float32))
        if skip_retain_grad_test:
            return
        assert y0.grad is None
        assert y1.grad is None

    def test_multiple_output_cpu(self):
        self.check_multiple_output_1arg(np)
        self.check_multiple_output_2args(np)

    @attr.gpu
    def test_multiple_output_gpu(self):
        self.check_multiple_output_1arg(cuda.cupy)
        self.check_multiple_output_2args(cuda.cupy)

    @attr.chainerx
    def test_multiple_output_chainerx_partially_ok(self):
        self.check_multiple_output_1arg(
            chainerx, skip_retain_grad_test=True)
        self.check_multiple_output_2args(
            chainerx, skip_retain_grad_test=True)

    # TODO(kataoka): Variable.backward with ChainerX backend unexpectedly
    # behaves like retain_grad=True
    @pytest.mark.xfail(strict=True)
    @attr.chainerx
    def test_multiple_output_1arg_chainerx(self):
        self.check_multiple_output_1arg(chainerx)

    # TODO(kataoka): Variable.backward with ChainerX backend unexpectedly
    # behaves like retain_grad=True
    @pytest.mark.xfail(strict=True)
    @attr.chainerx
    def test_multiple_output_2args_chainerx(self):
        self.check_multiple_output_2args(chainerx)

    def test_multiple_output_call_count(self):
        x = chainer.Variable(np.array([1, 2], np.float32))

        f = chainer.FunctionNode()
        f.forward = mock.MagicMock(
            side_effect=lambda xs: tuple(x * 2 for x in xs))
        f.backward = mock.MagicMock(
            side_effect=lambda _, gys: tuple(gy * 2 for gy in gys))

        h, = f.apply((x,))
        y0 = h * 3
        y1 = h * 4
        y0.grad = np.array([1, 10], np.float32)
        y1.grad = np.array([100, 1000], np.float32)
        chainer.backward([y0, y1])
        testing.assert_allclose(x.grad, np.array([806, 8060], np.float32))
        assert f.backward.call_count == 1

    def test_warn_no_grad(self):
        x = chainer.Variable(np.array(4, np.float32))
        x.grad = np.array(3, np.float32)
        y = x * 2
        with testing.assert_warns(RuntimeWarning):
            chainer.backward([y])
        testing.assert_allclose(x.grad, np.array(3, np.float32))
        assert y.grad is None

    def test_duplicate_outputs(self):
        x = chainer.Variable(np.array(0, np.float32))
        y = chainer.functions.identity(x)
        y.grad = np.array(3, np.float32)
        with testing.assert_warns(RuntimeWarning):
            chainer.backward([y, y])
        # 6 might be expected, but y.grad is used only once
        testing.assert_allclose(x.grad, np.array(3, np.float32))


# see also test_function_node.TestGradTypeCheck
class TestBackwardTypeCheck(unittest.TestCase):

    def _rand(self):
        return np.random.uniform(-1, 1, (2, 3)).astype(np.float32)

    def test_type_check(self):
        x = chainer.Variable(self._rand())
        y = x * x
        y.grad = self._rand()
        gy = chainer.Variable(self._rand())

        with self.assertRaises(TypeError):
            chainer.backward(y)
        with self.assertRaises(TypeError):
            chainer.backward([y], gy)

        chainer.backward([y])
        chainer.backward([y], [gy])


# see also test_function_node.TestGradValueCheck
class TestBackwardValueCheck(unittest.TestCase):

    def test_length_check(self):
        x = chainer.Variable(np.array(3, np.float32))
        y = chainer.functions.identity(x)
        gy = chainer.Variable(np.array(7, np.float32))

        with self.assertRaises(ValueError):
            chainer.backward([y], [])
        with self.assertRaises(ValueError):
            chainer.backward([y], [gy, gy])
        with self.assertRaises(ValueError):
            chainer.backward([], [gy])
        with self.assertRaises(ValueError):
            chainer.backward([y, y], [gy])

        chainer.backward([y], [gy])


testing.run_module(__name__, __file__)
