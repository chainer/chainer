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


def _triplet(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x, x



class TestConvolution3D(unittest.TestCase):

    def setUp(self):
        self.link = links.Convolution3D(3, 2, ksize=3, stride=2, pad=1)
        b = self.link.b.data
        b[...] = numpy.random.uniform(-1, 1, b.shape)
        self.link.zerograds()

        in_h, in_w, in_d = 4, 3, 3
        kh, kw, kd = _triplet(3)
        sh, sw, sd = _triplet(2)
        ph, pw, pd = _triplet(1)
        out_h = conv.get_conv_outsize(in_h, kh, sh, ph)
        out_w = conv.get_conv_outsize(in_w, kw, sw, pw)
        out_d = conv.get_conv_outsize(in_d, kd, sd, pd)

        self.x = numpy.random.uniform(-1, 1,
                                      (2, 3, in_h, in_w, in_d)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1,
                                       (2, 2, out_h, out_w, out_d)).astype(numpy.float32)


    def check_forward_consistency(self):
        x_cpu = chainer.Variable(self.x)
        y_cpu = self.link(x_cpu)
        self.assertEqual(y_cpu.data.dtype, numpy.float32)

        self.link.to_gpu()
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        y_gpu = self.link(x_gpu)
        self.assertEqual(y_gpu.data.dtype, numpy.float32)

        gradient_check.assert_allclose(y_cpu.data, y_gpu.data.get())

    @attr.cudnn
    @condition.retry(3)
    def test_forward_consistency(self):
        self.check_forward_consistency()


    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, (self.link.W, self.link.b), eps=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


    def check_pickling(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x)
        y_data1 = y.data

        del x, y

        pickled = pickle.dumps(self.link, -1)
        del self.link
        self.link = pickle.loads(pickled)

        x = chainer.Variable(x_data)
        y = self.link(x)
        y_data2 = y.data

        gradient_check.assert_allclose(y_data1, y_data2, atol=0, rtol=0)

    def test_pickling_cpu(self):
        self.check_pickling(self.x)

    @attr.gpu
    def test_pickling_gpu(self):
        self.link.to_gpu()
        self.check_pickling(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)
