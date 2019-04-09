import numpy

from chainer.functions import convolution_2d
from chainer.functions import deformable_convolution_2d_sampler
from chainer import utils

from chainer import testing


@testing.parameterize(*testing.product({
    'params': [
        (1, 1, 1, 1, 1, 1),
        (2, 2, 2, 2, 2, 2),
        (1, 2, 2, 1, 1, 2),
        (1, 2, 3, 4, 1, 2),
        (1, 2, 3, 4, 4, 5),
        (3, 3, 2, 2, 1, 1),
    ],
    'dtype': [numpy.float32],
    'use_cudnn': ['always', 'never']
}))
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['always'],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestDeformableConvolution2DSamplerFunctionZeroOffset(
        testing.FunctionTestCase):

    def setUp(self):
        self.skip_double_backward_test = True
        self.in_channels = 3
        self.out_channels = 2
        self.batch_size = 2
        self.h = 9
        self.w = 9

        self.kh, self.kw, self.sy, self.sx, self.ph, self.pw = self.params

        self.stride = (self.sy, self.sx)
        self.pad = (self.ph, self.pw)

        self.out_h = utils.conv.get_conv_outsize(
            self.h, self.kh, self.sy, self.ph)
        self.out_w = utils.conv.get_conv_outsize(
            self.w, self.kw, self.sx, self.pw)
        self.offset = numpy.zeros(
            (self.batch_size, 2 * self.kh * self.kw, self.out_h, self.out_w),
            dtype=self.dtype)

        self.check_backward_options = {'atol': 0.1, 'rtol': 1}

    def generate_inputs(self):
        W = numpy.random.normal(
            size=(self.out_channels, self.in_channels, self.kh, self.kw)
        ).astype(self.dtype)
        b = numpy.random.uniform(
            size=(self.out_channels,)).astype(self.dtype)

        x = numpy.random.uniform(
            size=(self.batch_size, self.in_channels, self.h, self.w)
        ).astype(self.dtype)

        offset = numpy.zeros(
            (self.batch_size, 2 * self.kh * self.kw, self.out_h, self.out_w)
        ).astype(self.dtype)

        return W, b, x, offset

    def forward_expected(self, inputs):
        W, b, x, _ = inputs
        y = convolution_2d(
            x, W, b, self.stride, self.pad)
        return y.array,

    def forward(self, inputs, devices):
        W, b, x, offset = inputs
        out = deformable_convolution_2d_sampler(
            x, offset, W, b, self.stride, self.pad)
        return out,


@testing.parameterize(*testing.product({
    'params': [
        (1, 1, 1, 1, 1, 1),
        (2, 2, 2, 2, 2, 2),
        (1, 2, 2, 1, 1, 2),
        (1, 2, 3, 4, 1, 2),
        (1, 2, 3, 4, 4, 5),
        (3, 3, 2, 2, 1, 1),
    ],
    'dtype': [numpy.float32],
    'use_cudnn': ['always', 'never']
}))
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['always'],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestDeformableConvolution2DSamplerFunctionLeftBottomOffset(
        testing.FunctionTestCase):

    def setUp(self):
        self.skip_double_backward_test = True
        self.in_channels = 3
        self.out_channels = 2
        self.batch_size = 2
        self.h = 9
        self.w = 9
        self.kh, self.kw, self.sy, self.sx, self.ph, self.pw = self.params

        self.stride = (self.sy, self.sx)
        self.pad = (self.ph, self.pw)

        self.pad = (
            self.pad[0] + 1 * self.stride[0], self.pad[1] + 1 * self.stride[1])

        self.out_h = utils.conv.get_conv_outsize(
            self.h, self.kh, self.sy, self.ph)
        self.out_w = utils.conv.get_conv_outsize(
            self.w, self.kw, self.sx, self.pw)

        self.check_backward_options = {'atol': 1, 'rtol': 5e-1}
        self.check_forward_options = {'atol': 10, 'rtol': 5e-1}

    def generate_inputs(self):
        W = numpy.random.normal(
            size=(self.out_channels, self.in_channels, self.kh, self.kw)
        ).astype(self.dtype)
        b = numpy.random.uniform(
            size=(self.out_channels,)).astype(self.dtype)
        x = numpy.random.uniform(
            size=(self.batch_size, self.in_channels, self.h, self.w)
        ).astype(self.dtype)

        offset = numpy.zeros(
            (self.batch_size, 2 * self.kh * self.kw, self.out_h, self.out_w),
            dtype=self.dtype)

        _, _, kh, kw = W.shape
        offset[:, :kh * kw] = -1 * self.stride[1]
        offset[:, kh * kw:] = 1 * self.stride[0]

        return W, b, x, offset

    def forward_expected(self, inputs):
        W, b, x, _ = inputs
        expected = convolution_2d(
            x, W, b, self.stride, self.pad).array
        expected = expected[:, :, 2:, :-2]
        return expected,

    def forward(self, inputs, device):
        W, b, x, offset = inputs
        out = deformable_convolution_2d_sampler(
            x, offset, W, b, self.stride, self.pad)
        return out,


testing.run_module(__name__, __file__)
