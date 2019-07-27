import numpy

from chainer import functions
from chainer import testing


@testing.parameterize(*(testing.product({
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'nobias': [True, False],
})))
@testing.inject_backend_tests(
    None,
    [{}]
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
    })
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
)
class TestDepthwiseConvolution2DFunction(testing.FunctionTestCase):

    def setUp(self):
        self.in_channels = 3
        self.channel_multiplier = 2
        self.kh, self.kw = (3, 3)
        self.stride = 2
        self.pad = 1
        self.check_forward_options.update({'atol': 5e-4, 'rtol': 5e-3})
        self.check_backward_options.update({'atol': 5e-4, 'rtol': 5e-3})
        self.check_double_backward_options.update({'atol': 5e-4, 'rtol': 5e-3})
        if numpy.float16 in (self.x_dtype, self.W_dtype):
            self.check_forward_options.update({'atol': 1e-3, 'rtol': 1e-2})
            self.check_backward_options.update({'atol': 1e-3, 'rtol': 1e-3})
            self.check_double_backward_options.update({
                'atol': 1e-3, 'rtol': 1e-3})

    def generate_inputs(self):
        x = numpy.random.uniform(
            -1, 1, (2, 3, 4, 3)).astype(self.x_dtype)
        W = numpy.random.normal(
            0, numpy.sqrt(1. / (self.kh * self.kw * self.in_channels)),
            (self.channel_multiplier, self.in_channels, self.kh, self.kw)
        ).astype(self.W_dtype)
        if self.nobias:
            return x, W
        b = numpy.random.uniform(
            -1, 1, self.in_channels * self.channel_multiplier
        ).astype(self.x_dtype)
        return x, W, b

    def forward(self, inputs, device):
        if self.nobias:
            x, W = inputs
            b = None
        else:
            x, W, b = inputs

        y = functions.depthwise_convolution_2d(
            x, W, b, stride=self.stride, pad=self.pad)
        y = sum(functions.split_axis(y, W.shape[1], axis=1))
        return y,

    def forward_expected(self, inputs):
        if self.nobias:
            x, W = inputs
            b = None
        else:
            x, W, b = inputs
            b = sum(numpy.split(b, W.shape[1]))
        y = functions.convolution_2d(
            x, W, b, stride=self.stride, pad=self.pad).array
        return y,


testing.run_module(__name__, __file__)
