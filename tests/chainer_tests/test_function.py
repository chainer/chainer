import unittest

import mock
import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


class TestFunction(unittest.TestCase):

    def _get_method(self, prefix, gpu):
        suffix = 'gpu' if gpu else 'cpu'
        return getattr(self.f, prefix + '_' + suffix)

    def setUp(self):
        y1 = numpy.arange(4).astype(numpy.float32)
        y2 = numpy.arange(4).astype(numpy.float32) + 1
        gx1 = numpy.arange(3).astype(numpy.float32)
        gx2 = None
        gy1 = numpy.arange(4).astype(numpy.float32)
        gy2 = numpy.arange(4).astype(numpy.float32)

        f = chainer.Function()
        f.check_type_forward = mock.MagicMock()
        f.forward_cpu = mock.MagicMock(return_value=(y1, y2))
        f.forward_gpu = mock.MagicMock()
        f.backward_cpu = mock.MagicMock(return_value=(gx1, gx2))
        f.backward_gpu = mock.MagicMock()
        self.f = f

        self.x1 = numpy.arange(3).astype(numpy.float32)
        self.x2 = numpy.arange(3).astype(numpy.int32)
        self.y1 = y1
        self.y2 = y2
        self.gx1 = gx1
        self.gx2 = gx2
        self.gy1 = gy1
        self.gy2 = gy2

    def tearDown(self):
        # Set None to delete cuda array
        self.f = None
        self.y1 = None
        self.y2 = None
        self.gx1 = None

    def setup_gpu(self):
        self.x1 = cuda.to_gpu(self.x1)
        self.x2 = cuda.to_gpu(self.x2)
        self.y1 = cuda.to_gpu(self.y1)
        self.y2 = cuda.to_gpu(self.y2)
        self.gx1 = cuda.to_gpu(self.gx1)
        self.gx2 = None
        self.gy1 = cuda.to_gpu(self.gy1)
        self.gy2 = cuda.to_gpu(self.gy2)
        self.f.forward_gpu = mock.MagicMock(return_value=(self.y1, self.y2))
        self.f.backward_gpu = mock.MagicMock(return_value=(self.gx1, self.gx2))

    def check_forward(self, gpu):
        y1, y2 = self.f.forward((self.x1, self.x2))
        self.assertEqual(self.f.check_type_forward.call_count, 0)
        self.assertEqual(self._get_method('forward', not gpu).call_count, 0)
        self._get_method('forward', gpu).assert_called_once_with(
            (self.x1, self.x2))
        self.assertTrue((cuda.to_cpu(y1) == cuda.to_cpu(self.y1)).all())
        self.assertTrue((cuda.to_cpu(y2) == cuda.to_cpu(self.y2)).all())

    def test_forward_cpu(self):
        self.check_forward(False)

    @attr.gpu
    def test_forward_gpu(self):
        self.setup_gpu()
        self.check_forward(True)

    def check_backward(self, gpu):
        gx1, gx2 = self.f.backward((self.x1, self.x2), (self.gy1, self.gy2))
        self.assertEqual(self._get_method('backward', not gpu).call_count, 0)
        self._get_method('backward', gpu).assert_called_once_with(
            (self.x1, self.x2), (self.gy1, self.gy2))
        self.assertTrue((cuda.to_cpu(gx1) == cuda.to_cpu(self.gx1)).all())
        self.assertIsNone(gx2)

    def test_backward_cpu(self):
        self.check_backward(False)

    @attr.gpu
    def test_backward_gpu(self):
        self.setup_gpu()
        self.check_backward(True)

    def check_check_type_forward(self):
        self.assertEqual(self.f.check_type_forward.call_count, 1)
        ts = self.f.check_type_forward.call_args[0][0]
        self.assertIsInstance(ts, type_check.TypeInfoTuple)
        self.assertEqual(len(ts), 2)

        self.assertEqual(ts[0].name, 'in_types[0]')
        t1 = ts[0].eval()
        self.assertEqual(t1.shape, (3,))
        self.assertEqual(t1.dtype, numpy.float32)

        self.assertEqual(ts[1].name, 'in_types[1]')
        t2 = ts[1].eval()
        self.assertEqual(t2.shape, (3,))
        self.assertEqual(t2.dtype, numpy.int32)

    def check_call(self):
        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2)
        x1.rank = 1
        x2.rank = 3
        ys = self.f(x1, x2)

        self.assertEqual(len(ys), 2)
        self.check_check_type_forward()

        for y in ys:
            self.assertIsInstance(y, chainer.Variable)
            # rank is (maximum rank in xs) + 1.
            self.assertEqual(y.rank, 4)
            self.assertFalse(y.volatile)
            # __call__ method does not make a copy anymore
            self.assertIs(y.creator, self.f)

        self.assertIsInstance(y.creator.outputs, tuple)

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.setup_gpu()
        self.check_call()

    def check_call_volatile(self):
        x1 = chainer.Variable(self.x1, volatile=True)
        x2 = chainer.Variable(self.x2, volatile=True)
        x1.rank = 1
        x2.rank = 3
        ys = self.f(x1, x2)

        self.assertEqual(len(ys), 2)
        self.check_check_type_forward()

        for y in ys:
            self.assertIsInstance(y, chainer.Variable)
            self.assertEqual(y.rank, 0)
            self.assertTrue(y.volatile)
            self.assertIsNone(y.creator)

    def test_call_volatile_cpu(self):
        self.check_call_volatile()

    @attr.gpu
    def test_call_volatile_gpu(self):
        self.setup_gpu()
        self.check_call_volatile()

    def check_call_single_return_value(self, volatile):
        x1 = chainer.Variable(self.x1, volatile=volatile)
        x2 = chainer.Variable(self.x2, volatile=volatile)
        ret = self.f(x1, x2)
        self.assertIsInstance(ret, chainer.Variable)

    def test_call_sigle_return_value_cpu(self):
        self.f.forward_cpu.return_value = (cuda.to_cpu(self.y1),)
        self.check_call_single_return_value(False)

    @attr.gpu
    def test_call_sigle_return_value_gpu(self):
        self.setup_gpu()
        self.f.forward_gpu.return_value = (cuda.to_gpu(self.y1),)
        self.check_call_single_return_value(False)

    def test_call_sigle_return_value_volatile_cpu(self):
        self.f.forward_cpu.return_value = (cuda.to_cpu(self.y1),)
        self.check_call_single_return_value(True)

    @attr.gpu
    def test_call_sigle_return_value_volatile_gpu(self):
        self.setup_gpu()
        self.f.forward_gpu.return_value = (cuda.to_gpu(self.y1),)
        self.check_call_single_return_value(True)

    def check_call_mixed_volatile(self):
        x1 = chainer.Variable(self.x1, volatile=True)
        x2 = chainer.Variable(self.x2, volatile=False)
        with self.assertRaises(AssertionError):
            self.f(x1, x2)

    def test_call_mixed_volatile_cpu(self):
        self.check_call_mixed_volatile()

    @attr.gpu
    def test_call_mixed_volatile_gpu(self):
        self.setup_gpu()
        self.check_call_mixed_volatile()

    def _get_f(self):
        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2)
        y1, y2 = self.f(x1, x2)

        f = y1.creator
        # To test weak refernece, return only x1 and y1.
        # x2 and y2 are deleted by the garbage collector
        return f, x1, y1

    def test_unchain(self):
        f, _x1, _y1 = self._get_f()
        f.unchain()

        y1, y2 = f.outputs
        # As _y1 is alive, this weak ref is also alive
        y1_ref = y1()
        self.assertTrue(y1_ref is not None and y1_ref.creator is None)
        # This weak ref is dead by unchain
        y2_ref = y2()
        self.assertTrue(y2_ref is None)

        self.assertIsNone(f.inputs)

    def test_label(self):
        self.assertEqual(self.f.label, 'Function')


class TestFunctionBackwardIntegration(unittest.TestCase):

    def test_backward(self):
        x = chainer.Variable(numpy.array([1]))
        y1 = F.identity(x)
        y2 = F.identity(x)
        z = y1 + y2

        z.grad = numpy.array([1])
        z.backward(retain_grad=True)

        self.assertEqual(y1.grad[0], 1)
        self.assertEqual(y2.grad[0], 1)
        self.assertEqual(x.grad[0], 2)


testing.run_module(__name__, __file__)
