import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'shape': [(3, 2), ()],
}))
class TestGaussian(unittest.TestCase):

    def setUp(self):
        self.m = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.v = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggm = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggv = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        self.check_double_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
        if self.dtype == numpy.float16:
            self.check_backward_options['dtype'] = numpy.float64
            self.check_double_backward_options['dtype'] = numpy.float64

    def check_forward(self, m_data, v_data):
        m = chainer.Variable(m_data)
        v = chainer.Variable(v_data)
        n = functions.gaussian(m, v)

        # Only checks dtype and shape because its result contains noise
        self.assertEqual(n.dtype, self.dtype)
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
        # TODO(hvy): Do no expose internals of the tested function using
        # e.g. numpy.random.RandomState
        gaussian = functions.noise.gaussian.Gaussian()

        def f(m, v):
            # In case numerical gradient computation is held in more precise
            # dtype than that of backward computation, cast the eps to reuse
            # before the numerical computation.
            if gaussian.eps is not None and gaussian.eps.dtype != m.dtype:
                gaussian.eps = gaussian.eps.astype(m.dtype)
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
        gaussian = functions.noise.gaussian.Gaussian()

        def f(m, v):
            if gaussian.eps is not None and gaussian.eps.dtype != m.dtype:
                gaussian.eps = gaussian.eps.astype(m.dtype)
            return gaussian.apply((m, v))

        gradient_check.check_double_backward(
            f, (m_data, v_data), y_grad, (m_grad_grad, v_grad_grad),
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.m, self.v, self.gy, self.ggm, self.ggv)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.m),
                                   cuda.to_gpu(self.v),
                                   cuda.to_gpu(self.gy),
                                   cuda.to_gpu(self.ggm),
                                   cuda.to_gpu(self.ggv))


@testing.parameterize(*testing.product({
    'specify_eps': [True, False],
}))
class TestGaussianEps(unittest.TestCase):

    def setUp(self):
        self.m = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.v = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.eps = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def _check(self):
        eps = self.eps if self.specify_eps else None
        out, out_eps = functions.gaussian(
            self.m, self.v, eps=eps, return_eps=True)
        assert isinstance(out_eps, type(out.array))
        if eps is None:
            assert out_eps.shape == out.array.shape
        else:
            assert out_eps is eps
        out2 = functions.gaussian(self.m, self.v, eps=out_eps)
        testing.assert_allclose(out.array, out2.array)

    def test_cpu(self):
        self._check()

    @attr.gpu
    def test_gpu(self):
        self.m = cuda.to_gpu(self.m)
        self.v = cuda.to_gpu(self.v)
        self.eps = cuda.to_gpu(self.eps)
        self._check()


testing.run_module(__name__, __file__)
