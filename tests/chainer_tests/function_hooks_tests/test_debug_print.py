import re
import unittest

import six

import chainer
from chainer.backends import cuda
from chainer import function_hooks
from chainer import testing
from chainer.testing import attr
import numpy


class DummyFunction(chainer.Function):

    def forward(self, inputs):
        self.retain_inputs((0,))
        return inputs[0],

    def backward(self, inputs, grads):
        return (grads[0],) + (None,) * (len(inputs) - 1)


class TestPrintHookToFunction(unittest.TestCase):

    def setUp(self):
        self.io = six.StringIO()
        self.h = function_hooks.PrintHook(file=self.io)
        self.f = DummyFunction()
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def test_forward_cpu(self):
        self.f.add_hook(self.h)
        self.f(chainer.Variable(self.x), chainer.Variable(self.x))
        # In some environments, shape is long.
        expect = r'''^function\tDummyFunction
input data
<variable at 0x[0-9a-f]+>
- device: CPU
- backend: <(type|class) 'numpy.ndarray'>
- shape: \(3L?, 5L?\)
- dtype: float32
- statistics: mean=[0-9.\-e]+, std=[0-9.\-e]+
- grad: None
<variable at 0x[0-9a-f]+>
- device: CPU
- backend: <(type|class) 'numpy.ndarray'>
- shape: \(3L?, 5L?\)
- dtype: float32
- statistics: mean=[0-9.\-e]+, std=[0-9.\-e]+
- grad: None$
'''
        actual = self.io.getvalue()
        self.assertTrue(re.match(expect, actual), actual)

    @attr.gpu
    def test_forward_gpu(self):
        self.f.add_hook(self.h)
        self.f(chainer.Variable(cuda.to_gpu(self.x)),
               chainer.Variable(cuda.to_gpu(self.x)))
        expect = r'''^function\tDummyFunction
input data
<variable at 0x[0-9a-f]+>
- device: <CUDA Device 0>
- backend: <(type|class) 'cupy.core.core.ndarray'>
- shape: \(3L?, 5L?\)
- dtype: float32
- statistics: mean=[0-9.\-e]+, std=[0-9.\-e]+
- grad: None
<variable at 0x[0-9a-f]+>
- device: <CUDA Device 0>
- backend: <(type|class) 'cupy.core.core.ndarray'>
- shape: \(3L?, 5L?\)
- dtype: float32
- statistics: mean=[0-9.\-e]+, std=[0-9.\-e]+
- grad: None$
'''
        actual = self.io.getvalue()
        self.assertTrue(re.match(expect, actual), actual)

    def test_backward_cpu(self):
        y = self.f(chainer.Variable(self.x), chainer.Variable(self.x))
        y.grad = self.gy
        self.f.add_hook(self.h)
        y.backward()
        expect = r'''^function\tDummyFunction
input data
<variable at 0x[0-9a-f]+>
- device: CPU
- backend: <(type|class) 'numpy.ndarray'>
- shape: \(3L?, 5L?\)
- dtype: float32
- statistics: mean=[0-9.\-e]+, std=[0-9.\-e]+
- grad: None
\(removed\)
output gradient
<variable at 0x[0-9a-f]+>
- device: CPU
- backend: <(type|class) 'numpy.ndarray'>
- shape: \(3L?, 5L?\)
- dtype: float32
- statistics: mean=[0-9.\-e]+, std=[0-9.\-e]+
- grad: mean=[0-9.\-e]+, std=[0-9.\-e]+$
'''
        actual = self.io.getvalue()
        self.assertTrue(re.match(expect, actual), actual)

    @attr.gpu
    def test_backward_gpu(self):
        y = self.f(chainer.Variable(cuda.to_gpu(self.x)),
                   chainer.Variable(cuda.to_gpu(self.x)))
        y.grad = cuda.to_gpu(self.gy)
        self.f.add_hook(self.h)
        y.backward()
        expect = r'''^function\tDummyFunction
input data
<variable at 0x[0-9a-f]+>
- device: <CUDA Device 0>
- backend: <(type|class) 'cupy.core.core.ndarray'>
- shape: \(3L?, 5L?\)
- dtype: float32
- statistics: mean=[0-9.\-e]+, std=[0-9.\-e]+
- grad: None
\(removed\)
output gradient
<variable at 0x[0-9a-f]+>
- device: <CUDA Device 0>
- backend: <(type|class) 'cupy.core.core.ndarray'>
- shape: \(3L?, 5L?\)
- dtype: float32
- statistics: mean=[0-9.\-e]+, std=[0-9.\-e]+
- grad: mean=[0-9.\-e]+, std=[0-9.\-e]+$
'''
        actual = self.io.getvalue()
        self.assertTrue(re.match(expect, actual), actual)


testing.run_module(__name__, __file__)
