import unittest

import numpy
import re
import six

import chainer
from chainer import cuda
from chainer import function_hooks
from chainer import functions
from chainer.functions.connection import linear
from chainer import links
from chainer import testing
from chainer.testing import attr


def check_history(self, t, function_type, used_bytes_type, acquired_bytes_type):
    self.assertIsInstance(t[0], function_type)
    self.assertIsInstance(t[1], used_bytes_type)
    self.assertIsInstance(t[2], acquired_bytes_type)


class TestCupyMemoryProfileHookHookToLink(unittest.TestCase):

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


class TestCupyMemoryProfileHookHookToFunction(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.CupyMemoryProfileHook()
        self.f = functions.Exp()
        self.f.add_hook(self.h)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

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


class TestCupyMemoryProfileSummary(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.CupyMemoryProfileHook()
        self.f = functions.Exp()
        self.f.add_hook(self.h)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    @attr.gpu
    def test_summary(self):
        x = cuda.to_gpu(self.x)
        self.f(chainer.Variable(x))
        self.f(chainer.Variable(x))
        self.assertEqual(2, len(self.h.call_history))
        self.assertEqual(1, len(self.h.summary()))

    @attr.gpu
    def test_print_report(self):
        x = cuda.to_gpu(self.x)
        self.f(chainer.Variable(x))
        self.f(chainer.Variable(x))
        io = six.StringIO()
        self.h.print_report(file=io)
        expect = '''\AFunctionName  UsedBytes  AcquiredBytes  Occurrence
 +Exp +[0-9.\-e]+.?B +[0-9.\-e]+.?B +[0-9]+$
'''
        actual = io.getvalue()
        self.assertTrue(re.match(expect, actual), actual)


testing.run_module(__name__, __file__)
