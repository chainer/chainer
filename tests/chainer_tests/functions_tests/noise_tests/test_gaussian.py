import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
}))
class TestGaussian(unittest.TestCase):

    def setUp(self):
        self.m = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.v = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.ggm = numpy.random.uniform(-1, 1, self.shape) \
            .astype(numpy.float32)
        self.ggv = numpy.random.uniform(-1, 1, self.shape) \
            .astype(numpy.float32)

        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, m_data, v_data):
        m = chainer.Variable(m_data)
        v = chainer.Variable(v_data)
        n = functions.gaussian(m, v)

        # Only checks dtype and shape because its result contains noise
        self.assertEqual(n.dtype, numpy.float32)
        self.assertEqual(n.shape, m.shape)

    def test_forward_cpu(self):
        self.check_forward(self.m, self.v)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.m), cuda.to_gpu(self.v))

    def check_backward(self, m_data, v_data, y_grad):
        # Instantiate the FunctionNode object outside the function that is
        # tested in order to reuse the same noise for the numerical gradient
        # computations (the noise is generated once during its first forward
        # pass, then reused)
        # TODO(hvy): Do no expose interals of the tested function using
        # e.g. numpy.random.RandomState
        gaussian = functions.Gaussian()

        def f(m, v):
            return gaussian.apply((m, v))[0]

        gradient_check.check_backward(
            f, (m_data, v_data), y_grad, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.m, self.v, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.m),
                            cuda.to_gpu(self.v),
                            cuda.to_gpu(self.gy))

    def check_double_backward(self, m_data, v_data, y_grad, m_grad_grad,
                              v_grad_grad):
        gaussian = functions.Gaussian()

        def f(m, v):
            y = gaussian.apply((m, v))[0]
            return y * y

        gradient_check.check_double_backward(
            f, (m_data, v_data), y_grad, (m_grad_grad, v_grad_grad),
            atol=5e-4, rtol=5e-3)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.m, self.v, self.gy, self.ggm, self.ggv)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.m),
                                   cuda.to_gpu(self.v),
                                   cuda.to_gpu(self.gy),
                                   cuda.to_gpu(self.ggm),
                                   cuda.to_gpu(self.ggv))


testing.run_module(__name__, __file__)
