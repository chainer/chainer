import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer.cuda import memory_pool
from chainer import function_hooks
from chainer import functions
from chainer.functions.connection import linear
from chainer import links
from chainer import testing
from chainer.testing import attr


def check_history(self, t, function_type, used_bytes_type,
                  acquired_bytes_type):
    self.assertIsInstance(t[0], function_type)
    self.assertIsInstance(t[1], used_bytes_type)
    self.assertIsInstance(t[2], acquired_bytes_type)


class TestCupyMemoryProfileHookToLink(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.CupyMemoryProfileHook()
        self.l = links.Linear(5, 5)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def test_name(self):
        self.assertEqual(self.h.name, 'CupyMemoryProfileHook')

    def check_forward(self, x):
        with self.h:
            self.l(chainer.Variable(x))
        self.assertEqual(1, len(self.h.call_history))
        check_history(self, self.h.call_history[0],
                      linear.LinearFunction, int, int)

    @attr.gpu
    def test_forward_gpu(self):
        self.l.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x, gy):
        x = chainer.Variable(x)
        y = self.l(x)
        y.grad = gy
        with self.h:
            y.backward()
        self.assertEqual(1, len(self.h.call_history))
        check_history(self, self.h.call_history[0],
                      linear.LinearFunction, int, int)

    @attr.gpu
    def test_backward_gpu(self):
        self.l.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestCupyMemoryProfileHookToFunction(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.CupyMemoryProfileHook()
        self.f = functions.Exp()
        self.f.add_hook(self.h)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def tearDown(self):
        self.f.delete_hook(self.h.name)

    def check_forward(self, x):
        self.f(chainer.Variable(x))
        self.assertEqual(1, len(self.h.call_history))
        check_history(self, self.h.call_history[0],
                      functions.Exp, int, int)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x, gy):
        x = chainer.Variable(x)
        y = self.f(x)
        y.grad = gy
        y.backward()
        self.assertEqual(2, len(self.h.call_history))
        check_history(self, self.h.call_history[1],
                      functions.Exp, int, int)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestCupyMemoryProfileReport(unittest.TestCase):

    def setUp(self):
        memory_pool.free_all_blocks()
        self.h = function_hooks.CupyMemoryProfileHook()
        self.f1 = functions.Exp()
        self.f2 = functions.Log()
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        x = cuda.to_gpu(self.x)
        with self.h:
            self.f1(chainer.Variable(x))
            self.f1(chainer.Variable(x))
            self.f2(chainer.Variable(x))
            self.f2(chainer.Variable(x))

    @attr.gpu
    def test_call_history(self):
        self.assertEqual(4, len(self.h.call_history))

    @attr.gpu
    def test_total_used_bytes(self):
        self.assertNotEqual(0, self.h.total_used_bytes())

    @attr.gpu
    def test_total_acquired_bytes(self):
        self.assertNotEqual(0, self.h.total_acquired_bytes())

    @attr.gpu
    def test_summary(self):
        self.assertEqual(2, len(self.h.summary()))

    @attr.gpu
    def test_print_report(self):
        io = six.StringIO()
        self.h.print_report(file=io)
        expect = r'''\AFunctionName  UsedBytes  AcquiredBytes  Occurrence
 +Exp +[0-9.\-e]+.?B +[0-9.\-e]+.?B +[0-9]+
 +Log +[0-9.\-e]+.?B +[0-9.\-e]+.?B +[0-9]+$
'''
        actual = io.getvalue()
        six.assertRegex(self, actual, expect)


testing.run_module(__name__, __file__)
