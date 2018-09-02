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
    'use_indices': [True, False],
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
            [1, 1, 6, 6],
            [6, 2, 7, 11],
            [3, 1, 5, 10],
            [3, 3, 3, 3]
        ], dtype=self.dtype)
        self.roi_indices = numpy.array(
            [0, 2, 1, 0], dtype=numpy.int32)
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

    def check_forward(self, x_data, roi_data, roi_index_data):
        x = chainer.Variable(x_data)
        if self.use_indices:
            rois = chainer.Variable(roi_data)
            roi_indices = chainer.Variable(roi_index_data)
            y = functions.roi_pooling_2d(
                x, rois, roi_indices=roi_indices,
                outh=self.outh, outw=self.outw,
                spatial_scale=self.spatial_scale)
        else:
            xp = cuda.get_array_module(roi_data)
            roi_data = xp.concatenate(
                (roi_index_data[:, None].astype(roi_data.dtype), roi_data),
                axis=1)
            rois = chainer.Variable(roi_data)
            y = functions.roi_pooling_2d(
                x, rois, outh=self.outh, outw=self.outw,
                spatial_scale=self.spatial_scale)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.rois, self.roi_indices)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.rois),
            cuda.to_gpu(self.roi_indices))

    @attr.gpu
    def test_forward_cpu_gpu_equal(self):
        # cpu
        x_cpu = chainer.Variable(self.x)
        if self.use_indices:
            rois_cpu = chainer.Variable(self.rois)
            roi_indices_cpu = chainer.Variable(self.roi_indices)
            y_cpu = functions.roi_pooling_2d(
                x_cpu, rois_cpu, roi_indices=roi_indices_cpu,
                outh=self.outh, outw=self.outw,
                spatial_scale=self.spatial_scale)
        else:
            roi_data = numpy.concatenate(
                (self.roi_indices[:, None].astype(self.rois.dtype), self.rois),
                axis=1)
            rois_cpu = chainer.Variable(roi_data)
            y_cpu = functions.roi_pooling_2d(
                x_cpu, rois_cpu,
                outh=self.outh, outw=self.outw,
                spatial_scale=self.spatial_scale)

        # gpu
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        if self.use_indices:
            rois_gpu = chainer.Variable(cuda.to_gpu(self.rois))
            roi_indices_gpu = chainer.Variable(
                cuda.to_gpu(self.roi_indices))
            y_gpu = functions.roi_pooling_2d(
                x_gpu, rois_gpu, roi_indices=roi_indices_gpu,
                outh=self.outh, outw=self.outw,
                spatial_scale=self.spatial_scale)
        else:
            roi_data = numpy.concatenate(
                (self.roi_indices[:, None].astype(self.rois.dtype), self.rois),
                axis=1)
            rois_gpu = chainer.Variable(cuda.to_gpu(roi_data))
            y_gpu = functions.roi_pooling_2d(
                x_gpu, rois_gpu,
                outh=self.outh, outw=self.outw,
                spatial_scale=self.spatial_scale)
        testing.assert_allclose(y_cpu.data, cuda.to_cpu(y_gpu.data))

    def check_backward(self, x_data, roi_data, roi_index_data, y_grad):
        def f(x, rois, roi_indices):
            if self.use_indices:
                return functions.roi_pooling_2d(
                    x, rois, roi_indices=roi_indices,
                    outh=self.outh, outw=self.outw,
                    spatial_scale=self.spatial_scale)
            else:
                xp = cuda.get_array_module(rois.array)
                roi_data = xp.concatenate(
                    (roi_indices.array[:, None].astype(rois.dtype),
                     rois.array), axis=1)
                rois = chainer.Variable(roi_data)
                return functions.roi_pooling_2d(
                    x, rois, outh=self.outh, outw=self.outw,
                    spatial_scale=self.spatial_scale)

        gradient_check.check_backward(
            f, (x_data, roi_data, roi_index_data),
            y_grad, no_grads=[False, True, True],
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.rois, self.roi_indices, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.rois),
            cuda.to_gpu(self.roi_indices), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
