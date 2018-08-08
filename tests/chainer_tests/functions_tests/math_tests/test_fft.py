import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape': [(4,), (2, 3), (2, 3, 2)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'method': ['fft', 'ifft']
}))
class TestFFT(unittest.TestCase):

    def setUp(self):
        self.rx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ix = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.rg = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ig = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.grx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.gix = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

    def check_forward(self, rx_data, ix_data):
        ry, iy = getattr(chainer.functions, self.method)((rx_data, ix_data))

        x = self.rx + self.ix * 1j
        y = getattr(numpy.fft, self.method)(x)

        testing.assert_allclose(y.real.astype(self.dtype), ry.data)
        testing.assert_allclose(y.imag.astype(self.dtype), iy.data)

    def test_forward_cpu(self):
        self.check_forward(self.rx, self.ix)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.rx), cuda.to_gpu(self.ix))

    def check_backward(self, rx_data, ix_data, rg_data, ig_data):
        def f(real, imag):
            return getattr(chainer.functions, self.method)((real, imag))

        gradient_check.check_backward(
            f, (rx_data, ix_data),
            (rg_data, ig_data), dtype='d', eps=2.0 ** -2, atol=1e-2, rtol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.rx, self.ix, self.rg, self.ig)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.rx), cuda.to_gpu(self.ix),
            cuda.to_gpu(self.rg), cuda.to_gpu(self.ig))

    def test_backward_real_cpu(self):
        self.check_backward(self.rx, self.ix, self.rg, None)

    def test_backward_imag_cpu(self):
        self.check_backward(self.rx, self.ix, None, self.ig)

    @attr.gpu
    def test_backward_real_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.rx), cuda.to_gpu(self.ix),
            cuda.to_gpu(self.rg), None)

    @attr.gpu
    def test_backward_imag_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.rx), cuda.to_gpu(self.ix),
            None, cuda.to_gpu(self.ig))

    def check_double_backward(self, rx, ix, rg, ig, grx, gix):
        def f(rx, ix):
            ry, iy = chainer.functions.fft((rx, ix))
            return ry * ry, iy * iy

        gradient_check.check_double_backward(
            f, (rx, ix), (rg, ig), (grx, gix), dtype='d', atol=1e-2, rtol=1e-3)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.rx, self.ix, self.rg, self.ig, self.grx, self.gix)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.rx), cuda.to_gpu(self.ix),
            cuda.to_gpu(self.rg), cuda.to_gpu(self.ig),
            cuda.to_gpu(self.grx), cuda.to_gpu(self.gix))


testing.run_module(__name__, __file__)
