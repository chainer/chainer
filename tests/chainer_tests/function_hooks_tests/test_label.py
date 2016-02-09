import unittest

import chainer
from chainer import links
from chainer import functions
from chainer import function_hooks
import numpy


class TestLabel(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.LabelHook()
        self.l = links.Linear(10, 2)
        self.f = functions.Exp()
        self.x = chainer.Variable(numpy.random.uniform(-1, 1, (3, 10)).astype(numpy.float32))

    def test_link_cpu(self):
        with self.h:
            self.l(self.x)

    def test_link_gpu(self):
        self.l.to_gpu()
        self.x.to_gpu()
        with self.h:
            self.l(self.x)

    def test_function_cpu(self):
        self.f.add_hook(self.h)
        self.f(self.x)

    def test_function_gpu(self):
        self.x.to_gpu()
        self.f(self.x)
