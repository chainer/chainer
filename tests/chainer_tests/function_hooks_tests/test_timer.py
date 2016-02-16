import unittest

import chainer
from chainer import cuda
from chainer import function_hooks
from chainer import functions
from chainer.functions.connection import linear
from chainer import gradient_check
from chainer import links
from chainer.testing import attr
import numpy


def check_history(self, t, propagation, process_type, function_type, return_type):
    self.assertTupleEqual((propagation, process_type), t[:2])
    self.assertIsInstance(t[2], function_type)
    self.assertIsInstance(t[3], return_type)


class TestTimerHookToLink(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.TimerHook()
        self.l = links.Linear(5, 5)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def check_forward(self, x):
        with self.h:
            self.l(chainer.Variable(x))
        self.assertEqual(2, len(self.h.hook_history))
        check_history(self, self.h.hook_history[0], 'forward', 'preprocess', linear.LinearFunction, type(None))
        check_history(self, self.h.hook_history[1], 'forward', 'postprocess', linear.LinearFunction, float)

    def test_forward_cpu(self):
        self.check_forward(self.x)

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
        self.assertEqual(2, len(self.h.hook_history))
        check_history(self, self.h.hook_history[0], 'backward', 'preprocess', linear.LinearFunction, type(None))
        check_history(self, self.h.hook_history[1], 'backward', 'postprocess', linear.LinearFunction, float)


    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.l.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.g))


class TestTimerHookToFunction(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.TimerHook()
        self.f = functions.Exp()
        self.f.add_hook(self.h)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def check_forward(self, x):
        self.f(chainer.Variable(x))
        self.assertEqual(2, len(self.h.hook_history))
        check_history(self, self.h.hook_history[0], 'forward', 'preprocess', functions.Exp, type(None))
        check_history(self, self.h.hook_history[1], 'forward', 'postprocess', functions.Exp, float)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_fowward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x, gy):
        x = chainer.Variable(x)
        y = self.f(x)
        y.grad = gy
        y.backward()
        self.assertEqual(4, len(self.h.hook_history))
        check_history(self, self.h.hook_history[2], 'backward', 'preprocess', functions.Exp, type(None))
        check_history(self, self.h.hook_history[3], 'backward', 'postprocess', functions.Exp, float)


    def test_backward_cpu(self):
        gradient_check.check_backward(self.f, self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        gradient_check.check_backward(
            self.f, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))
