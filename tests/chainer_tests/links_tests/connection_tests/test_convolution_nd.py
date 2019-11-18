import unittest

import numpy
import six.moves.cPickle as pickle

import chainer
from chainer.backends import cuda
from chainer import functions as F
from chainer import initializers
from chainer.links.connection import convolution_nd
from chainer import testing
from chainer.testing import attr
from chainer.utils import conv_nd


@testing.parameterize(*(testing.product({
    'dims': [(3, 4), (3, 4, 3)],
    'dtype': [numpy.float32],
    'in_channels': [4, None, 'omit'],
    'groups': [1, 2],
}) + testing.product({
    'dims': [(5,)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'in_channels': [4, None, 'omit'],
    'groups': [1, 2],
})))
@testing.inject_backend_tests(
    None,
    # CPU tests
    [{}]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestConvolutionND(testing.LinkTestCase):

    param_names = ('W', 'b')

    def setUp(self):
        self.ndim = len(self.dims)
        self.ksize = (3,) * self.ndim
        self.stride = (2,) * self.ndim
        self.pad = (1,) * self.ndim

        self.x_shape = (2, 4) + self.dims

        self.check_backward_options.update({'eps': 1e-2,
                                            'atol': 1e-3, 'rtol': 1e-3})
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 5e-3, 'rtol': 5e-2})
            self.check_backward_options.update({
                'eps': 2 ** -4, 'atol': 2 ** -4, 'rtol': 2 ** -4})

    def before_test(self, test_name):
        # cuDNN 5 and 5.1 results suffer from precision issues
        using_old_cudnn = (self.backend_config.xp is cuda.cupy
                           and self.backend_config.use_cudnn == 'always'
                           and cuda.cuda.cudnn.getVersion() < 6000)
        if using_old_cudnn:
            self.check_backward_options.update({
                'eps': 2 ** -4, 'atol': 2 ** -4, 'rtol': 2 ** -4})

    def generate_params(self):
        initial_bias = initializers.Uniform(scale=1., dtype=self.dtype)
        return initial_bias,

    def create_link(self, initializers):
        initial_bias, = initializers

        if self.in_channels == 'omit':
            link = convolution_nd.ConvolutionND(
                self.ndim, 2, self.ksize, stride=self.stride,
                pad=self.pad, groups=self.groups,
                initial_bias=initial_bias)
        else:
            link = convolution_nd.ConvolutionND(
                self.ndim, self.in_channels, 2, self.ksize, stride=self.stride,
                pad=self.pad, groups=self.groups,
                initial_bias=initial_bias)

        return link

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        return x,

    def forward_expected(self, link, inputs):
        x, = inputs
        W = link.W
        b = link.b
        y = F.convolution_nd(
            x, W, b,
            pad=self.pad,
            groups=self.groups,
            stride=self.stride)
        return y.array,

    def test_pickling(self, backend_config):
        x_data, = self.generate_inputs()

        link = self.create_link(self.generate_params())
        link.to_device(backend_config.device)

        x = chainer.Variable(x_data)
        x.to_device(backend_config.device)

        y = link(x)
        y_data1 = y.data
        del x, y
        pickled = pickle.dumps(link, -1)
        del link
        link = pickle.loads(pickled)
        x = chainer.Variable(x_data)
        x.to_device(backend_config.device)
        y = link(x)
        y_data2 = y.data

        testing.assert_allclose(y_data1, y_data2, atol=0, rtol=0)

    def test_from_params(self, backend_config):
        if (
                (backend_config.use_cuda and
                 backend_config.cuda_device == 1) or
                (backend_config.use_chainerx and
                 'cuda' in backend_config.chainerx_device)):
            raise unittest.SkipTest()
        link1 = self.create_link(self.generate_params())
        link1.to_device(backend_config.device)

        if self.in_channels in (None, 'omit'):
            link1._initialize_params(self.x_shape[1])

        link2 = convolution_nd.ConvolutionND.from_params(
            link1.W, link1.b,
            stride=self.stride, pad=self.pad, groups=self.groups)
        assert link2.W.shape == link1.W.shape
        assert link2.b.shape == link1.b.shape
        assert link2.stride == link1.stride
        assert link2.pad == link1.pad


@testing.parameterize(*(testing.product({
    'dims': [(3, 4), (3, 4, 3)],
    'dtype': [numpy.float32],
}) + testing.product({
    'dims': [(5,)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
class TestConvolutionNDIm2ColConsistency(unittest.TestCase):

    def setUp(self):
        self.ndim = len(self.dims)
        self.ksize = (3,) * self.ndim
        self.stride = (2,) * self.ndim
        self.pad = (1,) * self.ndim

        self.x_shape = (2, 4) + self.dims
        self.x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)

    @attr.gpu
    def test_im2col_consistency(self):
        col_cpu = conv_nd.im2col_nd_cpu(
            self.x, self.ksize, self.stride, self.pad)
        col_gpu = conv_nd.im2col_nd_gpu(
            cuda.to_gpu(self.x), self.ksize, self.stride, self.pad)
        testing.assert_allclose(col_cpu, col_gpu.get(), atol=0, rtol=0)

    @attr.gpu
    def test_col2im_consistency(self):
        col = conv_nd.im2col_nd_cpu(self.x, self.ksize, self.stride, self.pad)
        im_cpu = conv_nd.col2im_nd_cpu(col, self.stride, self.pad, self.dims)
        im_gpu = conv_nd.col2im_nd_gpu(
            cuda.to_gpu(col), self.stride, self.pad, self.dims)
        testing.assert_allclose(im_cpu, im_gpu.get())


class TestConvolutionNDNoInitialBias(unittest.TestCase):

    def test_no_initial_bias(self):
        ndim = 3
        ksize = 3
        link = convolution_nd.ConvolutionND(
            ndim, 3, 2, ksize, nobias=True)
        self.assertIsNone(link.b)


class TestConvolutionNDWrappers(unittest.TestCase):

    def _get_data(self, ndim):
        in_channels = 3
        out_channels = 2
        dtype = numpy.float32

        x_shape = (2, in_channels) + (3,) * ndim
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)

        return in_channels, out_channels, x

    def test_conv1d(self):
        in_c, out_c, x = self._get_data(1)
        link_nd = convolution_nd.ConvolutionND(1, in_c, out_c, 2, initialW=1)
        link_1d = convolution_nd.Convolution1D(in_c, out_c, 2, initialW=1)
        testing.assert_allclose(link_nd(x).data, link_1d(x).data)

    def test_conv3d(self):
        in_c, out_c, x = self._get_data(3)
        link_nd = convolution_nd.ConvolutionND(3, in_c, out_c, 2, initialW=1)
        link_3d = convolution_nd.Convolution3D(in_c, out_c, 2, initialW=1)
        testing.assert_allclose(link_nd(x).data, link_3d(x).data)


testing.run_module(__name__, __file__)
