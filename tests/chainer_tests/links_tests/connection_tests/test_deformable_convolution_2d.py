import unittest

import numpy

from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer import utils


@testing.parameterize(*testing.product({
    'nobias': [True, False],
    'initialization': ['explicit', 'placeholder']
}))
class TestDeformableConvolution2D(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 2
        batch_size = 2
        h = 9
        w = 9

        ksize = 3
        stride = 2
        pad = 1
        out_h = utils.conv.get_conv_outsize(h, ksize, stride, pad)
        out_w = utils.conv.get_conv_outsize(w, ksize, stride, pad)

        self.x = numpy.random.uniform(
            -1, 1, (batch_size, in_channels, h, w)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1,
            (batch_size, out_channels, out_h, out_w)).astype(numpy.float32)

        if self.initialization == 'explicit':
            pass
        elif self.initialization == 'placeholder':
            in_channels = None
        self.link = links.DeformableConvolution2D(
            in_channels, out_channels, ksize, stride=stride, pad=pad,
            offset_nobias=self.nobias, deform_nobias=self.nobias)

    def check_backward(self, x_data, y_grad):
        if self.nobias:
            params = (self.link.deform_conv.W)
        else:
            params = (self.link.deform_conv.W, self.link.deform_conv.b)

        gradient_check.check_backward(
            self.link, x_data,
            y_grad, params,
            eps=2 ** -3, atol=1e-3, rtol=1e-3)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
