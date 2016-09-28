import unittest

import numpy
import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
import chainer.functions as F
from chainer.functions.connection.attention import attention


@testing.parameterize(*testing.product({
}))
class TestAttention(unittest.TestCase):
    lengths = [3, 5, 4, 1]
    dim = 2
    batchsize = len(lengths)

    def setUp(self):
        q_shape = (self.batchsize, self.dim)
        self.q = numpy.random.uniform(-1, 1, q_shape).astype('f')
        self.xs = [numpy.random.uniform(-1, 1, (l, self.dim)).astype('f') for l in self.lengths]

        self.gy = numpy.random.uniform(-1, 1, q_shape).astype('f')

    def check_forward(self, q_data, xs_data):
        q = chainer.Variable(q_data)
        xs = [chainer.Variable(x) for x in xs_data]
        y = attention(q, xs)

        # expect
        scores = [F.softmax(F.transpose(F.matmul(x, qi))) for (x, qi) in zip(xs, q)]
        y_expect = F.concat([F.matmul(x, s) for (x, s) in zip(scores, xs)], axis=0)

        testing.assert_allclose(y_expect.data, y.data, atol=1e-3, rtol=1e-2)

    def test_forward_cpu(self):
        self.check_forward(self.q, self.xs,)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.q), [cuda.to_gpu(x) for x in self.xs])

    def check_backward(self, q_data, xs_data, gy_data):
        args = tuple([q_data,] + xs_data)
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
        self.check_backward(cuda.to_gpu(self.q), map(lambda x: cuda.to_gpu(x), self.xs), cuda.to_gpu(self.gy))

testing.run_module(__name__, __file__)