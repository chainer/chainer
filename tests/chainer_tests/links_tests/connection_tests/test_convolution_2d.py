import unittest

import numpy
import six.moves.cPickle as pickle

import chainer
from chainer.backends import cuda
from chainer import functions as F
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.utils import conv


@testing.parameterize(*testing.product({
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
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
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestConvolution2D(testing.LinkTestCase):

    param_names = ('W', 'b')

    skip_double_backward_test = True

    def setUp(self):
        self.N = 2
        self.in_channels = 3
        self.out_channels = 2
        self.ksize = 3
        self.stride = 2
        self.pad = 1
        if self.x_dtype == numpy.float16 or self.W_dtype == numpy.float16:
            self.check_forward_options.update({'atol': 5e-3, 'rtol': 5e-2})
            self.check_backward_options.update({'atol': 3e-2, 'rtol': 5e-2})

    def before_test(self, test_name):
        # cuDNN 5 and 5.1 results suffer from precision issues
        using_old_cudnn = (self.backend_config.xp is cuda.cupy
                           and self.backend_config.use_cudnn == 'always'
                           and cuda.cuda.cudnn.getVersion() < 6000)
        if using_old_cudnn:
            self.check_backward_options.update({'atol': 3e-2, 'rtol': 5e-2})

    def generate_params(self):
        initialW = chainer.initializers.Normal(1, self.W_dtype)
        initial_bias = chainer.initializers.Normal(1, self.x_dtype)
        return initialW, initial_bias

    def create_link(self, initializers):
        initialW, initial_bias = initializers

        link = links.Convolution2D(
            self.in_channels, self.out_channels, self.ksize,
            stride=self.stride, pad=self.pad,
            initialW=initialW,
            initial_bias=initial_bias)

        return link

    def generate_inputs(self):
        h, w = 4, 3
        x = numpy.random.uniform(-1, 1,
                                 (self.N, self.in_channels,
                                  h, w)).astype(self.x_dtype)
        return x,

    def forward_expected(self, link, inputs):
        x, = inputs
        W = link.W
        b = link.b
        y = F.convolution_2d(
            x, W, b,
            pad=self.pad,
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


@testing.parameterize(*testing.product({
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestConvolution2DIm2ColConsistency(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1,
                                      (2, 3, 4, 3)).astype(self.x_dtype)

    @attr.gpu
    def test_im2col_consistency(self):
        col_cpu = conv.im2col_cpu(self.x, 3, 3, 2, 2, 1, 1)
        col_gpu = conv.im2col_gpu(cuda.to_gpu(self.x), 3, 3, 2, 2, 1, 1)
        testing.assert_allclose(col_cpu, col_gpu.get(), atol=0, rtol=0)

    @attr.gpu
    def test_col2im_consistency(self):
        col = conv.im2col_cpu(self.x, 3, 3, 2, 2, 1, 1)
        h, w = self.x.shape[2:]
        im_cpu = conv.col2im_cpu(col, 2, 2, 1, 1, h, w)
        im_gpu = conv.col2im_gpu(cuda.to_gpu(col), 2, 2, 1, 1, h, w)
        testing.assert_allclose(im_cpu, im_gpu.get())


@testing.parameterize(*testing.product({
    'conv_args': [((None, 2, 3, 2, 1), {}),
                  ((2, 3), {'stride': 2, 'pad': 1})],
}))
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
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestConvolution2DParameterShapePlaceholder(testing.LinkTestCase):

    param_names = ('W', 'b')

    skip_double_backward_test = True

    def before_test(self, test_name):
        # cuDNN 5 and 5.1 results suffer from precision issues
        using_old_cudnn = (self.backend_config.xp is cuda.cupy
                           and self.backend_config.use_cudnn == 'always'
                           and cuda.cuda.cudnn.getVersion() < 6000)
        if using_old_cudnn:
            self.check_backward_options.update({'atol': 3e-2, 'rtol': 5e-2})

    def generate_params(self):
        return ()

    def create_link(self, initializers):

        args, kwargs = self.conv_args
        link = links.Convolution2D(*args, **kwargs)
        b = link.b.data
        b[...] = numpy.random.uniform(-1, 1, b.shape)

        return link

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1,
                                 (2, 3, 4, 3)).astype(numpy.float32)
        return x,

    def forward_expected(self, link, inputs):
        x, = inputs
        y = link(x).array
        return y,

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


testing.run_module(__name__, __file__)
