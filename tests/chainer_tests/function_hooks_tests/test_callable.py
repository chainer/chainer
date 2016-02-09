import unittest

import chainer
from chainer import links
from chainer import functions
from chainer import function_hooks
import numpy


def print_(function, in_data):
    print('test')

class TestCallable(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.CallableHook(print_)
        self.l = links.Linear(10, 2)
        self.f = functions.Exp()
        self.x = chainer.Variable(numpy.random.uniform(-1, 1, (3, 10)).astype(numpy.float32))

    def test_link_cpu(self):
        with self.h:
            self.l(self.x)

    def test_link_gpu(self):
        self.x.to_gpu()
        self.l.to_gpu()
        with self.h:
            self.l(self.x)

    def test_function_cpu(self):
        self.f.add_hook(self.h)
        self.f(self.x)

    def test_function_gpu(self):
        self.f.add_hook(self.h)
        self.x.to_gpu()
        self.f(self.x)
