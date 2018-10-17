import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import initializers
from chainer.links.connection import deconvolution_nd
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.testing import parameterize
from chainer.utils import conv


@parameterize(*testing.product({
    'dims': [(3, 2), (2,)],
    'nobias': [True, False],
    'dtype': [numpy.float32],
    'use_cudnn': ['always', 'auto', 'never'],
    'used_outsize': ['case1', 'case2', 'None'],
    'in_channels': [4, None, 'omit'],
    'groups': [1, 2],
}) + testing.product({
    'dims': [(4, 3, 2)],
    'nobias': [False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'use_cudnn': ['always'],
    'used_outsize': ['None'],
    'in_channels': [4, None, 'omit'],
    'groups': [1, 2],
}))
class TestDeconvolutionND(unittest.TestCase):

    def setUp(self):
        N = 2
        out_channels = 2
        ndim = len(self.dims)
        ksize = (3,) * ndim
        stride = (2,) * ndim
        pad = (1,) * ndim

        if self.used_outsize == 'case1' or self.used_outsize == 'None':
            # Use output size determined with get_deconv_outsize.
            outs = tuple(
                conv.get_deconv_outsize(d, k, s, p)
                for (d, k, s, p) in zip(self.dims, ksize, stride, pad))
        elif self.used_outsize == 'case2':
            # Use possible output size other than the one determined with
            # get_deconv_outsize.
            outs = tuple(
                conv.get_deconv_outsize(d, k, s, p) + 1
                for (d, k, s, p) in zip(self.dims, ksize, stride, pad))

        if self.used_outsize != 'None':
            outsize = outs
        else:
            outsize = None

        if not self.nobias:
            initial_bias = initializers.Uniform(scale=1, dtype=self.dtype)
        else:
            initial_bias = None

        if self.in_channels == 'omit':
            self.link = deconvolution_nd.DeconvolutionND(
                ndim, out_channels, ksize, stride=stride, pad=pad,
                outsize=outsize, initial_bias=initial_bias, nobias=self.nobias,
                groups=self.groups)
        else:
            self.link = deconvolution_nd.DeconvolutionND(
                ndim, self.in_channels, out_channels, ksize, stride=stride,
                pad=pad, outsize=outsize, initial_bias=initial_bias,
                nobias=self.nobias, groups=self.groups)
        self.link.cleargrads()

        x_shape = (N, 4) + self.dims
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        gy_shape = (N, out_channels) + outs
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(self.dtype)

        self.check_forward_options = {}
        self.check_backward_options = {
            'eps': 1e-2, 'atol': 1e-4, 'rtol': 1e-3}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-3, 'rtol': 5e-2}
            self.check_backward_options = {
                'eps': 2 ** -3, 'atol': 1e-2, 'rtol': 1e-1}

    def check_forward_consistency(self, link, x_data):
        x_cpu = chainer.Variable(x_data)
        y_cpu = link(x_cpu)
        self.assertEqual(y_cpu.data.dtype, x_data.dtype)

        link.to_gpu()
        x_gpu = chainer.Variable(cuda.to_gpu(x_data))
        y_gpu = link(x_gpu)
        self.assertEqual(y_gpu.data.dtype, x_data.dtype)

        testing.assert_allclose(
            y_cpu.data, y_gpu.data, **self.check_forward_options)

    @attr.gpu
    @condition.retry(3)
    def test_forward_consistency(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.check_forward_consistency(self.link, self.x)

    def check_backward(self, link, x_data, y_grad):
        params = [link.W]
        if not self.nobias:
            params.append(link.b)

        gradient_check.check_backward(
            link, x_data, y_grad, params, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.link, self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.check_backward(
                self.link, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestDeconvolutionNDNoInitialBias(unittest.TestCase):

    def test_no_initial_bias(self):
        ndim = 3
        ksize = 3
        link = deconvolution_nd.DeconvolutionND(
            ndim, 3, 2, ksize, nobias=True)
        self.assertIsNone(link.b)


class TestDeconvolutionNDWrappers(unittest.TestCase):

    def _get_data(self, ndim):
        in_channels = 3
        out_channels = 2
        dtype = numpy.float32

        x_shape = (2, in_channels) + (3,) * ndim
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)

        return in_channels, out_channels, x

    def test_deconv1d(self):
        in_c, out_c, x = self._get_data(1)
        link_nd = deconvolution_nd.DeconvolutionND(
            1, in_c, out_c, 2, initialW=1)
        link_1d = deconvolution_nd.Deconvolution1D(
            in_c, out_c, 2, initialW=1)
        testing.assert_allclose(link_nd(x).data, link_1d(x).data)

    def test_deconv3d(self):
        in_c, out_c, x = self._get_data(3)
        link_nd = deconvolution_nd.DeconvolutionND(
            3, in_c, out_c, 2, initialW=1)
        link_3d = deconvolution_nd.Deconvolution3D(
            in_c, out_c, 2, initialW=1)
        testing.assert_allclose(link_nd(x).data, link_3d(x).data)


testing.run_module(__name__, __file__)
