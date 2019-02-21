import unittest

import numpy

import chainer
from chainer import functions
from chainer import gradient_check
from chainer import testing


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'shape': [(3, 2), ()],
}))
@testing.backend.inject_backend_tests(
    None,
    [
        # NumPy
        {},
        # CuPy
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
        # ChainerX
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestGaussian(unittest.TestCase):

    def setUp(self):
        shape = self.shape
        dtype = self.dtype
        self.m = numpy.random.uniform(-1, 1, shape).astype(dtype)
        self.v = numpy.random.uniform(-1, 1, shape).astype(dtype)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
        self.ggm = numpy.random.uniform(-1, 1, shape).astype(dtype)
        self.ggv = numpy.random.uniform(-1, 1, shape).astype(dtype)

        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        self.check_double_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
        if self.dtype == numpy.float16:
            self.check_backward_options['dtype'] = numpy.float64
            self.check_double_backward_options['dtype'] = numpy.float64

    def test_forward(self, backend_config):
        m_data, v_data = backend_config.get_array((self.m, self.v))

        m = chainer.Variable(m_data)
        v = chainer.Variable(v_data)

        # Call forward without eps and retrieve it
        n1, eps = functions.gaussian(m, v, return_eps=True)

        self.assertIsInstance(eps, backend_config.xp.ndarray)
        self.assertEqual(n1.dtype, self.dtype)
        self.assertEqual(n1.shape, m.shape)
        self.assertEqual(eps.dtype, self.dtype)
        self.assertEqual(eps.shape, m.shape)

        # Call again with retrieved eps
        n2 = functions.gaussian(m, v, eps=eps)
        self.assertEqual(n2.dtype, self.dtype)
        self.assertEqual(n2.shape, m.shape)
        testing.assert_allclose(n1.array, n2.array)

    def test_backward(self, backend_config):
        m_data, v_data = backend_config.get_array((self.m, self.v))
        y_grad = backend_config.get_array(self.gy)
        eps = backend_config.get_array(
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype))

        def f(m, v):
            # In case numerical gradient computation is held in more precise
            # dtype than that of backward computation, cast the eps to reuse
            # before the numerical computation.
            eps_ = eps.astype(m.dtype)
            return functions.gaussian(m, v, eps=eps_)

        gradient_check.check_backward(
            f, (m_data, v_data), y_grad, **self.check_backward_options)

    def test_double_backward(self, backend_config):
        m_data, v_data = backend_config.get_array((self.m, self.v))
        y_grad = backend_config.get_array(self.gy)
        m_grad_grad, v_grad_grad = (
            backend_config.get_array((self.ggm, self.ggv)))
        eps = backend_config.get_array(
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype))

        def f(m, v):
            # In case numerical gradient computation is held in more precise
            # dtype than that of backward computation, cast the eps to reuse
            # before the numerical computation.
            eps_ = eps.astype(m.dtype)
            return functions.gaussian(m, v, eps=eps_)

        gradient_check.check_double_backward(
            f, (m_data, v_data), y_grad, (m_grad_grad, v_grad_grad),
            **self.check_double_backward_options)


testing.run_module(__name__, __file__)
