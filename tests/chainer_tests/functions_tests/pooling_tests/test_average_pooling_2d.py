import unittest

import mock
import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product_dict(
    [
        {'pad': 1, 'ksize': 3,
         'in_shape': (2, 3, 4, 3), 'out_shape': (2, 3, 2, 2),
         'x_ranges': [(0, 2), (1, 4)], 'y_ranges': [(0, 2), (1, 3)]},
        {'pad': -1, 'ksize': 2,
         'in_shape': (2, 3, 5, 4), 'out_shape': (2, 3, 1, 1),
         'x_ranges': [(1, 3)], 'y_ranges': [(1, 3)]},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
class TestAveragePooling2D(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1,
                                      self.in_shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1,
                                       self.out_shape).astype(self.dtype)
        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.average_pooling_2d(x, self.ksize, stride=2,
                                         pad=self.pad, use_cudnn=use_cudnn)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        for k in six.moves.range(2):
            for c in six.moves.range(3):
                x = self.x[k, c]
                expect = numpy.array(
                    [[x[rx1:rx2, ry1:ry2].sum() for ry1, ry2 in self.y_ranges]
                     for rx1, rx2 in self.x_ranges]) / (self.ksize ** 2)
                testing.assert_allclose(
                    expect, y_data[k, c], **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), False)

    def check_backward(self, x_data, y_grad, use_cudnn=True):
        gradient_check.check_backward(
            functions.AveragePooling2D(
                self.ksize, 2, self.pad, False, use_cudnn),
            x_data, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), False)


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestAveragePooling2DCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.arange(
            2 * 3 * 4 * 3, dtype=self.dtype).reshape(2, 3, 4, 3)
        self.gy = cuda.cupy.random.uniform(-1, 1,
                                           (2, 3, 2, 2)).astype(self.dtype)

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.average_pooling_2d(
            x, 3, stride=2, pad=1, use_cudnn=self.use_cudnn)

    @unittest.skipIf(cuda.cudnn_enabled and
                     cuda.cudnn.cudnn.getVersion() < 3000,
                     'Only cudnn ver>=3 supports average-pooling2d')
    def test_call_cudnn_forward(self):
        with mock.patch('cupy.cudnn.cudnn.poolingForward') as func:
            self.forward()
            self.assertEqual(func.called, self.use_cudnn)

    @unittest.skipIf(cuda.cudnn_enabled and
                     cuda.cudnn.cudnn.getVersion() < 3000,
                     'Only cudnn ver>=3 supports average-pooling2d')
    def test_call_cudnn_backward(self):
        y = self.forward()
        y.grad = self.gy
        with mock.patch('cupy.cudnn.cudnn.poolingBackward') as func:
            y.backward()
            self.assertEqual(func.called, self.use_cudnn)


testing.run_module(__name__, __file__)
