import unittest

import mock
import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import links
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'use_cudnn': True},
    {'use_cudnn': False},
)
class TestSeparableConvolution2D(unittest.TestCase):

    def setUp(self):
        depthwise_conv = links.Convolution2D(
            3, 2, 3, stride=1, pad=1,
            use_cudnn=self.use_cudnn)
        pointwise_conv = links.Convolution2D(
            3, 2, 3, stride=1, pad=1,
            use_cudnn=self.use_cudnn)
        self.sconv2d = links.SeparableConvolution2D(
            depthwise_conv=depthwise_conv,
            pointwise_conv=pointwise_conv)
        self.x = numpy.zeros((10, 3, 20, 20), dtype=numpy.float32)

    def test_init(self):
        self.assertEqual(len(self.sconv2d), 3)
        for i, conv in enumerate(self.sconv2d):
            self.assertIsInstance(conv, links.Convolution2D)
            self.assertEqual(conv.use_cudnn, self.use_cudnn)
            if i == 0:
                self.assertEqual(conv.W.data.shape, (96, 3, 11, 11))
            else:
                self.assertEqual(conv.W.data.shape, (96, 96, 1, 1))

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        actual = self.sconv2d(x)
        x = self.sconv2d[0](x)
        for layer in self.link[1:]:
            ys.append(layer(x))
        expect = numpy.concatenate(ys)
        numpy.testing.assert_array_equal(
            cuda.to_cpu(expect.data), cuda.to_cpu(actual.data))

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.sconv2d.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = self.link(x)
        y.grad = y_grad
        y.backward()

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

@testing.parameterize(
    {'use_cudnn': True},
    {'use_cudnn': False},
)
@attr.cudnn
class TestSeparableConvolution2DCudnnCall(unittest.TestCase):

    def setUp(self):
        self.sconv2d = links.SeparableConvolution2D(
            3, (96, 96, 96), 11,
            activation=functions.sigmoid,
            use_cudnn=self.use_cudnn)
        self.sconv2d.to_gpu()
        self.x = cuda.cupy.zeros((10, 3, 20, 20), dtype=numpy.float32)
        self.gy = cuda.cupy.zeros((10, 96, 10, 10), dtype=numpy.float32)

    def forward(self):
        x = chainer.Variable(self.x)
        return self.sconv2d(x)

    def test_call_cudnn_forward(self):
        with mock.patch('cupy.cudnn.cudnn.convolutionForward') as func:
            self.forward()
            self.assertEqual(func.called, self.use_cudnn)

    def test_call_cudnn_backrward(self):
        y = self.forward()
        print(y.data.shape)
        y.grad = self.gy
        v2 = 'cupy.cudnn.cudnn.convolutionBackwardData_v2'
        v3 = 'cupy.cudnn.cudnn.convolutionBackwardData_v3'
        with mock.patch(v2) as func_v2,  mock.patch(v3) as func_v3:
            y.backward()
            self.assertEqual(func_v2.called or func_v3.called, self.use_cudnn)


@testing.parameterize(
    {'use_cudnn': True},
    {'use_cudnn': False},
)
class TestSeparableConvolution2DShapePlaceholder(unittest.TestCase):

    def setUp(self):
        self.sconv2d = links.SeparableConvolution2D(
            None, (96, 96, 96), 11,
            activation=functions.sigmoid,
            use_cudnn=self.use_cudnn)
        self.x = numpy.zeros((10, 3, 20, 20), dtype=numpy.float32)

    def test_init(self):
        self.assertIs(self.sconv2d.activation, functions.sigmoid)
        self.assertEqual(len(self.sconv2d), 3)

    def check_call(self, x_data):
        x = chainer.Variable(x_data)
        actual = self.sconv2d(x)
        act = functions.sigmoid
        expect = self.sconv2d[2](act(self.sconv2d[1](act(self.sconv2d[0](x)))))
        numpy.testing.assert_array_equal(
            cuda.to_cpu(expect.data), cuda.to_cpu(actual.data))
        for i, conv in enumerate(self.sconv2d):
            self.assertIsInstance(conv, links.Convolution2D)
            self.assertEqual(conv.use_cudnn, self.use_cudnn)
            if i == 0:
                self.assertEqual(conv.W.data.shape, (96, 3, 11, 11))
            else:
                self.assertEqual(conv.W.data.shape, (96, 96, 1, 1))

    def test_call_cpu(self):
        self.check_call(self.x)

    @attr.gpu
    def test_call_gpu(self):
        self.sconv2d.to_gpu()
        self.check_call(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)
