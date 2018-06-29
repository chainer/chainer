import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import links
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        {'in_size': 10, 'hidden_num': 1, 'hidden_size': 30},
        {'in_size': 10, 'hidden_num': 3, 'hidden_size': 10},
    ],
    [
        {'input_variable': False},
        {'input_variable': True},
    ]
))
class TestMADE(unittest.TestCase):

    def setUp(self):
        self.link = links.MADE(self.in_size, self.hidden_num, self.hidden_size)
        self.link.cleargrads()
        x_shape = (4, self.in_size)
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data) if self.input_variable else x_data
        y = self.link(x)
        self.assertEqual(y.shape, x.shape)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_autoregressive(self, x_data):
        m0 = self.link.m0
        order = m0.argsort()
        for i in range(self.in_size):
            x2_data = x_data.copy()
            change = order[i:]
            x2_data[:, change] = self.link.xp.random.uniform(
                -1, 1, x2_data[:, change].shape).astype(numpy.float32)
            x = chainer.Variable(x_data) if self.input_variable else x_data
            x2 = chainer.Variable(x2_data) if self.input_variable else x2_data
            y = self.link(x)
            y2 = self.link(x2)
            testing.assert_allclose(
                y[:, order[:i+1]].data, y2[:, order[:i+1]].data)

    def test_autoregressive_cpu(self):
        self.check_autoregressive(self.x)

    @attr.gpu
    def test_autoregressive_gpu(self):
        self.link.to_gpu()
        self.check_autoregressive(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)
