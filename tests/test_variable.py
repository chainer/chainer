import unittest

import numpy as np

import chainer
from chainer import cuda
from chainer.testing import attr

import six

if cuda.available:
    cuda.init()


class Constant(chainer.Function):

    def __init__(self, outputs):
        self.outputs = outputs

    def forward_cpu(self, inputs):
        return self.outputs

    def forward_gpu(self, inputs):
        return tuple(map(cuda.to_gpu, self.outputs))

    def backward_cpu(self, inputs, grad_outputs):
        return tuple(map(np.zeros_like, inputs))

    def backward_gpu(self, inputs, grad_outputs):
        return tuple(map(cuda.zeros_like, inputs))


def constant(xs, value):
    return Constant(value)(*xs)


class TestVariable(unittest.TestCase):

    def setUp(self):
        self.x = np.random.uniform(-1, 1, 10).astype(np.float32)
        self.a = np.random.uniform(0.1, 10, 10).astype(np.float32)
        self.c = np.arange(10).reshape(2, 5).astype(np.float32)

    def check_len(self, gpu):
        x = self.x
        if gpu:
            x = cuda.to_gpu(x)
        x = chainer.Variable(x)
        self.assertEqual(len(x), 10)

        x2 = chainer.Variable(x)
        self.assertEqual(len(x2), 10)

    def test_len_cpu(self):
        self.check_len(False)

    @attr.gpu
    def test_len_gpu(self):
        self.check_len(True)

    def check_label(self, expected, gpu):
        c = self.c
        if gpu:
            c = cuda.to_gpu(c)
        c = chainer.Variable(c)
        self.assertEqual(c.label, expected)

    def test_label_cpu(self):
        self.check_label('(2, 5), float32', False)

    @attr.gpu
    def test_label_gpu(self):
        self.check_label('(2, 5), float32', True)

    @attr.gpu
    def test_to_gpu_roundtrip(self):
        v = chainer.Variable(np.array([1]))
        result = v.to_gpu()
        self.assertTrue(isinstance(result.data, cuda.GPUArray))
        result = v.to_cpu()
        self.assertTrue(isinstance(result.data, np.ndarray))

    def check_backward(self, inputs, intermediates, outputs, retain_grad):
        for o in outputs:
            o.backward(retain_grad)

        self.assertTrue(all([x.grad is not None for x in inputs]))
        if retain_grad:
            self.assertTrue(all([x.grad is not None for x in intermediates]))
        else:
            self.assertTrue(all([x.grad is None for x in intermediates]))
        self.assertTrue(any([x.grad is not None for x in outputs]))

    # length is number of edges. So, # of Variables created is length+1
    def create_linear_chain(self, length, gpu):
        if gpu:
            x = chainer.Variable(cuda.to_gpu(self.x))
        else:
            x = chainer.Variable(self.x)
        ret = [x]
        for i in six.moves.range(length):
            ret.append(constant((ret[i], ), (self.a, )))
        ret[-1].grad = np.zeros_like(ret[-1].data)
        return ret

    def test_backward_cpu(self):
        ret = self.create_linear_chain(2, False)
        self.check_backward((ret[0], ), (ret[1], ), (ret[2], ), False)

    @attr.gpu
    def test_backward_gpu(self):
        ret = self.create_linear_chain(2, True)
        self.check_backward((ret[0], ), (ret[1], ), (ret[2], ), False)

    def test_backward_cpu_retain_grad(self):
        ret = self.create_linear_chain(2, False)
        self.check_backward((ret[0], ), (ret[1], ), (ret[2], ), True)

    @attr.gpu
    def test_backward_gpu_retain_grad(self):
        ret = self.create_linear_chain(2, True)
        self.check_backward((ret[0], ), (ret[1], ), (ret[2], ), True)

    def test_unchain_backward_cpu(self):
        ret = self.create_linear_chain(3, False)
        ret[1].unchain_backward()
        self.check_backward((ret[1], ), (ret[2], ), (ret[3], ), False)

    @attr.gpu
    def test_unchain_backward_gpu(self):
        ret = self.create_linear_chain(3, True)
        ret[1].unchain_backward()
        self.check_backward((ret[1], ), (ret[2], ), (ret[3], ), False)

    def test_unchain_backward_cpu_retain_grad(self):
        ret = self.create_linear_chain(3, False)
        ret[1].unchain_backward()
        self.check_backward((ret[1], ), (ret[2], ), (ret[3], ), False)

    @attr.gpu
    def test_unchain_backward_gpu_retain_grad(self):
        ret = self.create_linear_chain(3, False)
        ret[1].unchain_backward()
        self.check_backward((ret[1], ), (ret[2], ), (ret[3], ), False)

    def test_invalid_value_type(self):
        msg = "'data' must be numpy.ndarray, cuda.GPUArray or Variable"
        with self.assertRaisesRegexp(ValueError, msg):
            chainer.Variable(1)

        msg = "'volatile' must be bool type"
        with self.assertRaisesRegexp(ValueError, msg):
            chainer.Variable(np.array([1]), volatile='INVALID')
