import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import links as L
from chainer import testing
from chainer.testing import parameterize


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


@parameterize(
    *testing.product({
        'nobias': [True, False],
        'dilate': [1, 2],
        'groups': [1, 3],
        'x_dtype': [numpy.float32],
        'W_dtype': [numpy.float32],
    })
)
@testing.inject_backend_tests(
    ['test_forward', 'test_backward'],
    # CPU tests
    [
        {},
        {'use_ideep': 'always'},
    ]
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
class TestDeconvolution2D(testing.LinkTestCase):

    def setUp(self):
        self.in_channels = 3
        self.out_channels = 6
        self.ksize = 3
        self.stride = 2
        self.pad = 1
        if self.nobias:
            TestDeconvolution2D.param_names = ('W',)
        else:
            TestDeconvolution2D.param_names = ('W', 'b')

        self.check_backward_options.update({'atol': 1e-3, 'rtol': 1e-2})

    def before_test(self, test_name):
        # cuDNN 5 and 5.1 results suffer from precision issues
        using_old_cudnn = (self.backend_config.xp is cuda.cupy
                           and self.backend_config.use_cudnn == 'always'
                           and cuda.cuda.cudnn.getVersion() < 6000)
        if using_old_cudnn:
            self.check_backward_options.update({'atol': 3e-2, 'rtol': 5e-2})

    def generate_inputs(self):
        N = 2
        h, w = 3, 2
        x = numpy.random.uniform(
            -1, 1, (N, self.in_channels, h, w)).astype(self.x_dtype)
        return x,

    def generate_params(self):
        initialW = chainer.initializers.Normal(1, self.W_dtype)
        initial_bias = chainer.initializers.Normal(1, self.x_dtype)
        return initialW, initial_bias

    def create_link(self, initializers):
        initialW, initial_bias = initializers

        if self.nobias:
            link = L.Deconvolution2D(
                self.in_channels, self.out_channels, self.ksize,
                stride=self.stride, pad=self.pad, nobias=self.nobias,
                dilate=self.dilate, groups=self.groups,
                initialW=initialW)
        else:
            link = L.Deconvolution2D(
                self.in_channels, self.out_channels, self.ksize,
                stride=self.stride, pad=self.pad, nobias=self.nobias,
                dilate=self.dilate, groups=self.groups,
                initialW=initialW,
                initial_bias=initial_bias)

        return link

    def forward_expected(self, link, inputs):
        x, = inputs
        W = link.W
        if self.nobias:
            y = F.deconvolution_2d(
                x, W,
                stride=self.stride, pad=self.pad,
                dilate=self.dilate, groups=self.groups)
        else:
            b = link.b
            y = F.deconvolution_2d(
                x, W, b,
                stride=self.stride, pad=self.pad,
                dilate=self.dilate, groups=self.groups)
        return y.array,


@parameterize(
    *testing.product({
        'nobias': [True, False],
        'use_cudnn': ['always', 'never'],
        'deconv_args': [((3, 2, 3), {}), ((2, 3), {}), ((None, 2, 3), {}),
                        ((2, 3), {'stride': 2, 'pad': 1}),
                        ((None, 2, 3, 2, 1), {})]
    })
)
@testing.inject_backend_tests(
    ['test_forward', 'test_backward'],
    # CPU tests
    [
        {},
        {'use_ideep': 'always'},
    ]
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
class TestDeconvolution2DParameterShapePlaceholder(testing.LinkTestCase):

    def setUp(self):
        if self.nobias:
            self.param_names = ('W',)
        else:
            self.param_names = ('W', 'b')
        self.check_backward_options.update({'atol': 1e-4, 'rtol': 1e-3})

    def before_test(self, test_name):
        # cuDNN 5 and 5.1 results suffer from precision issues
        using_old_cudnn = (self.backend_config.xp is cuda.cupy
                           and self.backend_config.use_cudnn == 'always'
                           and cuda.cuda.cudnn.getVersion() < 6000)
        if using_old_cudnn:
            self.check_backward_options.update({'atol': 3e-2, 'rtol': 5e-2})

    def generate_inputs(self):
        N = 2
        h, w = 3, 2
        x = numpy.random.uniform(
            -1, 1, (N, 3, h, w)).astype(numpy.float32)
        return x,

    def generate_params(self):
        return []

    def create_link(self, initializers):

        args, kwargs = self.deconv_args
        kwargs['nobias'] = self.nobias
        link = L.Deconvolution2D(*args, **kwargs)
        if not self.nobias:
            link.b.data[...] = numpy.random.uniform(
                -1, 1, link.b.data.shape).astype(numpy.float32)

        return link

    def forward_expected(self, link, inputs):
        x, = inputs
        y = link(x).array
        return y,


testing.run_module(__name__, __file__)
