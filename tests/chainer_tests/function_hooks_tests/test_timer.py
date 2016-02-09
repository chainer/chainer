import unittest

import chainer
from chainer import cuda
from chainer import function_hooks
from chainer import functions
from chainer import gradient_check
from chainer import links
import numpy


class TestTimerHookToLink(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.TimerHook()
        self.l = links.Linear(5, 5)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def test_forward_cpu(self):
        with self.h:
            self.l(chainer.Variable(self.x))

    def test_forward_gpu(self):
        self.l.to_gpu()
        with self.h:
            self.l(chainer.Variable(cuda.to_gpu(self.x)))

    def test_backward_cpu(self):
        with self.h:
            gradient_check.check_backward(self.l, self.x, self.gy)

    def test_backward_gpu(self):
        self.l.to_gpu()
        with self.h:
            gradient_check.check_backward(
                self.l, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestTimerHookToFunction(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.TimerHook()
        self.f = functions.Exp()
        self.f.add_hook(self.h)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def test_forward_cpu(self):
        self.f(chainer.Variable(self.x))

    def test_fowward_gpu(self):
        self.f(chainer.Variable(cuda.to_gpu(self.x)))

    def test_backward_cpu(self):
        gradient_check.check_backward(self.f, self.x, self.gy)

    def test_backward_gpu(self):
        gradient_check.check_backward(
            self.f, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))
