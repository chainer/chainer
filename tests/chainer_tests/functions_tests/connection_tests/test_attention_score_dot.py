import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions import attention_score_dot
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'lengths': [[3, 5, 4, 1], [1, 1]],
    'dim': [3, 10],
}))
class TestAttentionScoreDot(unittest.TestCase):

    def setUp(self):
        batchsize = len(self.lengths)
        q_shape = (batchsize, self.dim)
        self.q = numpy.random.uniform(-1, 1, q_shape).astype('f')
        self.xs = [
            numpy.random.uniform(-1, 1, (l, self.dim)).astype('f')
            for l in self.lengths]

        self.gys = [numpy.random.uniform(-1, 1, (l,)).astype('f')
                    for l in self.lengths]

    def check_forward(self, q_data, xs_data):
        q = chainer.Variable(q_data)
        xs = [chainer.Variable(x) for x in xs_data]
        scores = attention_score_dot(q, xs)

        # expect
        scores_expect = [F.reshape(F.softmax(F.transpose(
            F.matmul(x, qi))), (-1,)) for (x, qi) in zip(xs, q)]
        for si, si_expect in zip(scores, scores_expect):
            testing.assert_allclose(
                si.data, si_expect.data, atol=1e-3, rtol=1e-2)

    def test_forward_cpu(self):
        self.check_forward(self.q, self.xs,)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.q), [
                           cuda.to_gpu(x) for x in self.xs])

    def check_backward(self, q_data, xs_data, gys_data):
        args = tuple([q_data, ] + xs_data)
        grads = gys_data

        def f(*inputs):
            q = inputs[0]
            xs = inputs[1:]
            y = attention_score_dot(q, xs)
            return y

        gradient_check.check_backward(
            f, args, grads, eps=1e-2, rtol=1e-3, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.q, self.xs, self.gys)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.q),
                            map(lambda x: cuda.to_gpu(x), self.xs),
                            map(lambda gy: cuda.to_gpu(gy), self.gys))

testing.run_module(__name__, __file__)
