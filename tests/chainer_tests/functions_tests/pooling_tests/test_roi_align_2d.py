import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'sampling_ratio': [0, 1, 2, (1, 2)],
    'use_indices': [True, False],
}))
class TestROIAlign2D(unittest.TestCase):

    def setUp(self):
        N = 3
        n_channels = 3
        self.x = numpy.arange(
            N * n_channels * 12 * 8,
            dtype=numpy.float32).reshape((N, n_channels, 12, 8))
        numpy.random.shuffle(self.x)
        self.x = 2 * self.x / self.x.size - 1
        self.rois = numpy.array([
            [1, 1, 6, 6],
            [6, 2, 7, 11],
            [3, 1, 5, 10],
            [3, 3, 3, 3],
            [1.1, 2.2, 3.3, 4.4],
        ], dtype=numpy.float32)
        self.roi_indices = numpy.array(
            [0, 2, 1, 0, 2], dtype=numpy.int32)
        n_rois = self.rois.shape[0]
        self.outh, self.outw = 5, 7
        self.spatial_scale = 0.6
        self.gy = numpy.random.uniform(
            -1, 1, (n_rois, n_channels,
                    self.outh, self.outw)).astype(numpy.float32)
        self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data, roi_data, roi_index_data):
        x = chainer.Variable(x_data)
        if self.use_indices:
            rois = chainer.Variable(roi_data)
            roi_indices = chainer.Variable(roi_index_data)
            y = functions.roi_average_align_2d(
                x, rois, roi_indices=roi_indices,
                outh=self.outh, outw=self.outw,
                spatial_scale=self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
            )
        else:
            xp = cuda.get_array_module(roi_data)
            roi_data = xp.concatenate(
                (roi_index_data[:, None].astype(roi_data.dtype), roi_data),
                axis=1)
            rois = chainer.Variable(roi_data)
            y = functions.roi_average_align_2d(
                x, rois, outh=self.outh, outw=self.outw,
                spatial_scale=self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
            )
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.rois, self.roi_indices)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.rois),
            cuda.to_gpu(self.roi_indices))

    @attr.gpu
    @condition.retry(3)
    def test_forward_cpu_gpu_equal(self):
        # cpu
        x_cpu = chainer.Variable(self.x)
        if self.use_indices:
            rois_cpu = chainer.Variable(self.rois)
            roi_indices_cpu = chainer.Variable(self.roi_indices)
            y_cpu = functions.roi_average_align_2d(
                x_cpu, rois_cpu, roi_indices=roi_indices_cpu,
                outh=self.outh, outw=self.outw,
                spatial_scale=self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
            )
        else:
            roi_data = numpy.concatenate(
                (self.roi_indices[:, None].astype(self.rois.dtype), self.rois),
                axis=1)
            rois_cpu = chainer.Variable(roi_data)
            y_cpu = functions.roi_average_align_2d(
                x_cpu, rois_cpu,
                outh=self.outh, outw=self.outw,
                spatial_scale=self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
            )

        # gpu
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        if self.use_indices:
            rois_gpu = chainer.Variable(cuda.to_gpu(self.rois))
            roi_indices_gpu = chainer.Variable(
                cuda.to_gpu(self.roi_indices))
            y_gpu = functions.roi_average_align_2d(
                x_gpu, rois_gpu, roi_indices=roi_indices_gpu,
                outh=self.outh, outw=self.outw,
                spatial_scale=self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
            )
        else:
            roi_data = numpy.concatenate(
                (self.roi_indices[:, None].astype(self.rois.dtype), self.rois),
                axis=1)
            rois_gpu = chainer.Variable(cuda.to_gpu(roi_data))
            y_gpu = functions.roi_average_align_2d(
                x_gpu, rois_gpu, outh=self.outh, outw=self.outw,
                spatial_scale=self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
            )
        testing.assert_allclose(y_cpu.data, cuda.to_cpu(y_gpu.data))

    def check_backward(self, x_data, roi_data, roi_index_data, y_grad):
        def f(x, rois, roi_indices):
            if self.use_indices:
                return functions.roi_average_align_2d(
                    x, rois, roi_indices=roi_indices,
                    outh=self.outh, outw=self.outw,
                    spatial_scale=self.spatial_scale,
                    sampling_ratio=self.sampling_ratio)
            else:
                xp = cuda.get_array_module(rois.array)
                roi_data = xp.concatenate(
                    (roi_indices.array[:, None].astype(rois.dtype),
                     rois.array), axis=1)
                rois = chainer.Variable(roi_data)
                return functions.roi_average_align_2d(
                    x, rois, outh=self.outh, outw=self.outw,
                    spatial_scale=self.spatial_scale,
                    sampling_ratio=self.sampling_ratio)

        gradient_check.check_backward(
            f, (x_data, roi_data, roi_index_data),
            y_grad, no_grads=[False, True, True],
            **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.rois, self.roi_indices, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.rois),
            cuda.to_gpu(self.roi_indices), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
