import unittest

import mock
import numpy

import chainer
from chainer.backends import cuda
from chainer import function_hooks
from chainer import testing
from chainer.testing import attr


@attr.gpu
@unittest.skipUnless(
    cuda.available and cuda.cupy.cuda.nvtx_enabled, 'nvtx is not installed')
class TestCUDAProfileHook(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.CUDAProfileHook()
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype('f')
        self.gy = numpy.random.uniform(-1, 1, (2, 3)).astype('f')

    def test_name(self):
        self.assertEqual(self.h.name, 'CUDAProfileHook')

    def check_forward(self, x):
        with mock.patch('cupy.cuda.nvtx.RangePush') as push, \
                mock.patch('cupy.cuda.nvtx.RangePop') as pop:
            with self.h:
                chainer.Variable(x) + chainer.Variable(x)

        push.assert_called_once_with('_ + _.forward')
        pop.assert_called_once_with()

    def test_forward_cpu(self):
        self.check_forward(self.x)

    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x, gy):
        y = chainer.Variable(x) + chainer.Variable(x)
        y.grad = gy
        with mock.patch('cupy.cuda.nvtx.RangePush') as push, \
                mock.patch('cupy.cuda.nvtx.RangePop') as pop:
            with self.h:
                y.backward()

        push.assert_called_once_with('_ + _.backward')
        pop.assert_called_once_with()

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


@attr.gpu
class TestCUDAProfileHookNVTXUnavailable(unittest.TestCase):

    def setUp(self):
        self.nvtx_enabled = cuda.cupy.cuda.nvtx_enabled
        cuda.cupy.cuda.nvtx_enabled = False

    def tearDown(self):
        cuda.cupy.cuda.nvtx_enabled = self.nvtx_enabled

    def test_unavailable(self):
        with self.assertRaises(RuntimeError):
            function_hooks.CUDAProfileHook()


testing.run_module(__name__, __file__)
