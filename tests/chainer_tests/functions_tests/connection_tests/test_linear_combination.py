import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions.connection.attention import attention
from chainer.functions.connection.attention_score_dot \
    import attention_score_dot
from chainer.functions.connection.linear_combination \
    import linear_combination
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
}))
class TestLinearCombination(unittest.TestCase):
    lengths = [5, 4, 2, 3, 1]
    dim = 10
    batchsize = len(lengths)

    def setUp(self):
        self.xs = [
            numpy.random.uniform(-1, 1, (l, self.dim)).astype('f')
            for l in self.lengths]
        self.cs = [numpy.random.uniform(-1, 1, (l, )).astype('f')
                   for l in self.lengths]
        self.gy = numpy.random.uniform(-1, 1,
                                       (self.batchsize, self.dim)).astype('f')

    def check_forward(self, xs_data, cs_data):
        xs = [chainer.Variable(x) for x in xs_data]
        cs = [chainer.Variable(c) for c in cs_data]
        y = linear_combination(xs, cs)

        # expect
        y_expect = [F.reshape(F.matmul(F.transpose(x), c), (-1,))
                    for (x, c) in zip(xs, cs)]
        for yi, yi_expect in zip(y, y_expect):
            testing.assert_allclose(
                yi.data, yi_expect.data, atol=1e-3, rtol=1e-2)

    def test_forward_cpu(self):
        self.check_forward(self.xs, self.cs)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(x) for x in self.xs], [
                           cuda.to_gpu(c) for c in self.cs])

    def check_backward(self, xs_data, cs_data, gy_data):
        args = tuple(xs_data + cs_data)
        grads = gy_data

        def f(*inputs):
            batchsize = len(inputs)
            xs = inputs[:batchsize]
            cs = inputs[batchsize:]
            y = linear_combination(xs, cs)
            return y

        gradient_check.check_backward(
            f, args, grads, eps=1e-2, rtol=1e-3, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.xs, self.cs, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(map(lambda x: cuda.to_gpu(x), self.xs),
                            map(lambda x: cuda.to_gpu(x), self.cs),
                            cuda.to_gpu(self.gy))


@testing.parameterize(*testing.product({
}))
class TestAttention(unittest.TestCase):
    lengths = [2, 3, 4, 2, 7]
    dim = 5
    batchsize = len(lengths)

    def setUp(self):
        q_shape = (self.batchsize, self.dim)
        self.q = numpy.random.uniform(-1, 1, q_shape).astype('f')
        self.xs = [
            numpy.random.uniform(-1, 1, (l, self.dim)).astype('f')
            for l in self.lengths]

        self.gy = numpy.random.uniform(-1, 1, q_shape).astype('f')

    def check_forward(self, q_data, xs_data):
        q = chainer.Variable(q_data)
        xs = [chainer.Variable(x) for x in xs_data]
        cs = attention_score_dot(q, xs)
        y = linear_combination(xs, cs)

        # expect
        y_expect = attention(q, xs)

        testing.assert_allclose(y_expect.data, y.data, atol=1e-3, rtol=1e-2)

    def test_forward_cpu(self):
        self.check_forward(self.q, self.xs,)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.q), [
                           cuda.to_gpu(x) for x in self.xs])

    def check_backward(self, q_data, xs_data, gy_data):
        args = tuple([q_data, ] + xs_data)
        grads = gy_data,

        def f(*inputs):
            q = inputs[0]
            xs = inputs[1:]
            y = attention(q, xs)
            return (y,)

        gradient_check.check_backward(
            f, args, grads, eps=1e-2, rtol=1e-3, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.q, self.xs, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.q), map(
            lambda x: cuda.to_gpu(x), self.xs), cuda.to_gpu(self.gy))

testing.run_module(__name__, __file__)
