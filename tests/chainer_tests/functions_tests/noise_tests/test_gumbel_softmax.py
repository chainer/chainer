import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestGumbelSoftmax(unittest.TestCase):

    def setUp(self):
        self.log_pi = numpy.random.uniform(
            -1, 1, self.shape).astype(numpy.float32)
        self.tau = numpy.float32(numpy.random.uniform(0.1, 10.0))

    def check_forward(self, log_pi_data, tau):
        log_pi = chainer.Variable(log_pi_data)
        y = functions.gumbel_softmax(log_pi, tau=tau)

        # Only checks dtype and shape because its result contains noise
        self.assertEqual(y.dtype, numpy.float32)
        self.assertEqual(y.shape, log_pi.shape)
        self.assertEqual(
            backend.get_array_module(y),
            backend.get_array_module(log_pi))

    def test_forward_cpu(self):
        self.check_forward(self.log_pi, self.tau)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.log_pi), self.tau)


testing.run_module(__name__, __file__)
