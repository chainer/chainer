import unittest

import numpy
import six.moves.cPickle as pickle

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import conv


@testing.parameterize(*testing.product({
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
}))
class TestConvolution2D(unittest.TestCase):

    def setUp(self):
        self.link = links.Convolution2D(
            3, 5, 3, stride=2, pad=1,
            initialW=chainer.initializers.Normal(1, self.W_dtype),
            initial_bias=chainer.initializers.Normal(1, self.x_dtype))
        self.link.cleargrads()

        self.x = numpy.random.uniform(-1, 1,
                                      (6, 3, 4, 3)).astype(self.x_dtype)
        self.gy = numpy.random.uniform(-1, 1,
                                       (6, 5, 2, 2)).astype(self.x_dtype)
        self.check_backward_options = {}

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, (self.link.W, self.link.b), eps=2 ** -3,
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x)
        print(y.data.shape)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

testing.run_module(__name__, __file__)
