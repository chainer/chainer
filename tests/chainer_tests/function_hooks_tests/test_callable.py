import unittest

import chainer
from chainer import cuda
from chainer import function_hooks
from chainer import functions
from chainer import links
from chainer.testing import attr
import numpy


class TestCallableHookToLink(unittest.TestCase):

    def c(self, function, in_data, out_grad):
        self.count += 1

    def setUp(self):
        self.h = function_hooks.CallableHook(lambda f, i, o: self.c(f, i, o))
        self.l = links.Linear(5, 5)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.count = 0

    def test_forward_cpu(self):
        with self.h:
            self.l(chainer.Variable(self.x))
        self.assertEqual(self.count, 2)

    @attr.gpu
    def test_forward_gpu(self):
        self.l.to_gpu()
        with self.h:
            self.l(chainer.Variable(cuda.to_gpu(self.x)))
        self.assertEqual(self.count, 2)

    def check_backward(self, x, gy):
        x = chainer.Variable(x)
        y = self.l(x)
        y.grad = gy
        with self.h:
            y.backward()
        self.assertEqual(self.count, 2)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.l.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestCallableHookToFunction(unittest.TestCase):

    def c(self, function, in_data, out_grad):
        self.count += 1

    def setUp(self):
        self.h = function_hooks.CallableHook(lambda f, i, o: self.c(f, i, o))
        self.f = functions.Exp()
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.count = 0

    def check_forward(self, x):
        x = chainer.Variable(x)
        self.f.add_hook(self.h)
        self.f(x)
        self.assertEqual(self.count, 2)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_fowward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x, gy):
        x = chainer.Variable(x)
        y = self.f(x)
        y.grad = gy
        self.f.add_hook(self.h)
        y.backward()
        self.assertEqual(self.count, 2)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))
