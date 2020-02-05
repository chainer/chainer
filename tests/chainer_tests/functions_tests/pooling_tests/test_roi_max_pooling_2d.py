import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer_tests.functions_tests.pooling_tests import pooling_nd_helper


def _pair(x):
    if isinstance(x, chainer.utils.collections_abc.Iterable):
        return x
    return x, x


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'outsize': [
        5, 7, (5, 7),
        (numpy.int32(5), numpy.int32(7))],
    'spatial_scale': [
        0.6, 1.0, 2, numpy.float32(0.6), numpy.int32(2)],
}))
class TestROIMaxPooling2D(unittest.TestCase):

    def setUp(self):
        N = 3
        n_channels = 3
        self.x = pooling_nd_helper.shuffled_linspace(
            (N, n_channels, 12, 8), self.dtype)
        self.rois = numpy.array([
            [1, 1, 7, 7],
            [2, 6, 12, 8],
            [1, 3, 11, 6],
            [3, 3, 4, 4]
        ], dtype=self.dtype)
        self.roi_indices = numpy.array([0, 2, 1, 0], dtype=numpy.int32)
        n_rois = self.rois.shape[0]
        outsize = _pair(self.outsize)
        self.gy = numpy.random.uniform(
            -1, 1, (n_rois, n_channels,
                    outsize[0], outsize[1])).astype(self.dtype)
        if self.dtype == numpy.float16:
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
        else:
            self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x_data, roi_data, roi_index_data):
        x = chainer.Variable(x_data)
        rois = chainer.Variable(roi_data)
        roi_indices = chainer.Variable(roi_index_data)
        y = functions.roi_max_pooling_2d(
            x, rois, roi_indices, outsize=self.outsize,
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
        rois_cpu = chainer.Variable(self.rois)
        roi_indices_cpu = chainer.Variable(self.roi_indices)
        y_cpu = functions.roi_max_pooling_2d(
            x_cpu, rois_cpu, roi_indices_cpu, outsize=self.outsize,
            spatial_scale=self.spatial_scale)

        # gpu
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        rois_gpu = chainer.Variable(cuda.to_gpu(self.rois))
        roi_indices_gpu = chainer.Variable(cuda.to_gpu(self.roi_indices))
        y_gpu = functions.roi_max_pooling_2d(
            x_gpu, rois_gpu, roi_indices_gpu, outsize=self.outsize,
            spatial_scale=self.spatial_scale)
        testing.assert_allclose(y_cpu.data, cuda.to_cpu(y_gpu.data))

    def check_backward(self, x_data, roi_data, roi_index_data, y_grad):
        def f(x, rois, roi_indices):
            y = functions.roi_max_pooling_2d(
                x, rois, roi_indices, outsize=self.outsize,
                spatial_scale=self.spatial_scale)
            xp = cuda.get_array_module(y)
            # replace -inf with zero for gradient_check
            y = functions.where(
                xp.isinf(y.array), xp.zeros(y.shape, dtype=y.dtype), y)
            return y

        gradient_check.check_backward(
            f, (x_data, roi_data, roi_index_data), y_grad,
            no_grads=[False, True, True], **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.rois, self.roi_indices, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.rois),
            cuda.to_gpu(self.roi_indices), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
