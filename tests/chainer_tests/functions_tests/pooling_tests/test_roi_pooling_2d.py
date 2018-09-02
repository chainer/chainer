import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'order': ['xy', 'yx'],
}))
class TestROIPooling2D(unittest.TestCase):

    def setUp(self):
        N = 3
        n_channels = 3
        self.x = numpy.arange(
            N * n_channels * 12 * 8,
            dtype=numpy.float32).reshape((N, n_channels, 12, 8))
        numpy.random.shuffle(self.x)
        self.x = (2 * self.x / self.x.size - 1).astype(self.dtype)
        self.rois = numpy.array([
            [0, 1, 1, 6, 6],
            [2, 6, 2, 7, 11],
            [1, 3, 1, 5, 10],
            [0, 3, 3, 3, 3]
        ], dtype=self.dtype)
        n_rois = self.rois.shape[0]
        self.outh, self.outw = 5, 7
        self.spatial_scale = 0.6
        self.gy = numpy.random.uniform(
            -1, 1, (n_rois, n_channels,
                    self.outh, self.outw)).astype(self.dtype)
        if self.dtype == numpy.float16:
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
        else:
            self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

        if self.dtype == 'yx':
            self.rois = self.rois[:, [0, 2, 1, 4, 3]]

    def check_forward(self, x_data, roi_data):
        x = chainer.Variable(x_data)
        rois = chainer.Variable(roi_data)
        y = functions.roi_pooling_2d(
            x, rois, outh=self.outh, outw=self.outw,
            spatial_scale=self.spatial_scale, order=self.order)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.rois)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.rois))

    @attr.gpu
    def test_forward_cpu_gpu_equal(self):
        # cpu
        x_cpu = chainer.Variable(self.x)
        rois_cpu = chainer.Variable(self.rois)
        y_cpu = functions.roi_pooling_2d(
            x_cpu, rois_cpu, outh=self.outh, outw=self.outw,
            spatial_scale=self.spatial_scale, order=self.order)

        # gpu
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        rois_gpu = chainer.Variable(cuda.to_gpu(self.rois))
        y_gpu = functions.roi_pooling_2d(
            x_gpu, rois_gpu, outh=self.outh, outw=self.outw,
            spatial_scale=self.spatial_scale, order=self.order)
        testing.assert_allclose(y_cpu.data, cuda.to_cpu(y_gpu.data))

    def check_backward(self, x_data, roi_data, y_grad):
        def f(x, rois):
            return functions.roi_pooling_2d(
                x, rois, outh=self.outh, outw=self.outw,
                spatial_scale=self.spatial_scale, order=self.order)

        gradient_check.check_backward(
            f, (x_data, roi_data), y_grad, no_grads=[False, True],
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.rois, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.rois),
                            cuda.to_gpu(self.gy))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
}))
class TestROIPooling2DROIOrder(unittest.TestCase):

    def setUp(self):
        N = 3
        n_channels = 3
        self.x = numpy.arange(
            N * n_channels * 12 * 8,
            dtype=numpy.float32).reshape((N, n_channels, 12, 8))
        numpy.random.shuffle(self.x)
        self.x = (2 * self.x / self.x.size - 1).astype(self.dtype)
        self.roi_xy = numpy.array([
            [0, 1, 1, 6, 6],
            [2, 6, 2, 7, 11],
            [1, 3, 1, 5, 10],
            [0, 3, 3, 3, 3]
        ], dtype=self.dtype)
        self.roi_yx = self.roi_xy[:, [0, 2, 1, 4, 3]]
        self.outh, self.outw = 5, 7
        self.spatial_scale = 0.6

    def check_forward_roi_order(self, x_data, roi_xy_data, roi_yx_data):
        x = chainer.Variable(x_data)
        rois_xy = chainer.Variable(roi_xy_data)
        rois_yx = chainer.Variable(roi_yx_data)
        y_xy = functions.roi_pooling_2d(
            x, rois_xy, outh=self.outh, outw=self.outw,
            spatial_scale=self.spatial_scale, order='xy')
        y_yx = functions.roi_pooling_2d(
            x, rois_yx, outh=self.outh, outw=self.outw,
            spatial_scale=self.spatial_scale, order='yx')
        testing.assert_allclose(y_xy.data, y_yx.data)

    def test_forward_roi_order_cpu(self):
        self.check_forward_roi_order(self.x, self.roi_xy, self.roi_yx)

    @attr.gpu
    def test_forward_roi_order_gpu(self):
        self.check_forward_roi_order(
            cuda.to_gpu(self.x),
            cuda.to_gpu(self.roi_xy), cuda.to_gpu(self.roi_yx))


testing.run_module(__name__, __file__)
