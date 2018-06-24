import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'delta_v': [0.5],
    'delta_d': [5],
    'alpha': [1],
    'beta': [1],
    'gamma': [0.001],
    'norm': [1],
    'result': [9709.69497664]
}) + testing.product(({
    'delta_v': [3],
    'delta_d': [10],
    'alpha': [0.1],
    'beta': [0.1],
    'gamma': [0.1],
    'max_n_clusters': [2],
    'norm': [2],
    'result': [2140205.11050112]
})))
class TestDiscriminativeMarginBasedClusteringLoss(unittest.TestCase):

    def setUp(self):
        self.max_n_clusters = 5
        self.batch = 5
        self.width = 10
        self.height = 10
        shape = (self.batch, self.max_n_clusters,
                 self.height, self.width)

        input_arr = numpy.linspace(0, 100,
                                   shape[0] * shape[1] *
                                   shape[2] * shape[3])
        self.input = input_arr.reshape(shape)
        self.gt = numpy.linspace(100, 0,
                                 shape[0] * shape[1] *
                                 shape[2] * shape[3])
        self.gt = self.gt.reshape(shape)
        self.gt_obj = [1, 2, 3, 4, 5]
        self.gt_obj_idx = [[0], [0, 2],
                           [0, 3, 4], [0, 1, 2, 3],
                           [0, 1, 2, 3, 4]]
        self.y = numpy.asarray(self.result)

    def get_result(self, prediction, labels, n_objects, gt_idx):
        out = functions.discriminative_margin_based_clustering_loss(
            prediction, labels, n_objects, gt_idx,
            self.delta_v, self.delta_d, self.max_n_clusters,
            self.norm, self.alpha, self.beta, self.gamma)
        return out

    def check_forward_cpu(self, prediction, labels, n_objects, gt_idx, t_data):
        t = chainer.Variable(t_data)
        out = self.get_result(prediction, labels, n_objects, gt_idx)
        numpy.testing.assert_almost_equal(out.data, t.data)

    def check_forward_gpu(self, prediction, labels, n_objects, gt_idx, t_data):
        t = chainer.Variable(t_data)
        out = self.get_result(prediction, labels, n_objects, gt_idx)
        out.to_cpu()
        t.to_cpu()
        numpy.testing.assert_almost_equal(out.data, t.data)

    def test_forward_cpu(self):
        self.check_forward_cpu(cuda.to_cpu(self.input), cuda.to_cpu(self.gt),
                               self.gt_obj, self.gt_obj_idx, self.y)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward_gpu(cuda.to_gpu(self.input), cuda.to_gpu(self.gt),
                               self.gt_obj, self.gt_obj_idx, self.y)

    @attr.gpu
    def test_forward_gpu_cpu(self):
        cpu_res = self.get_result(cuda.to_cpu(self.input),
                                  cuda.to_cpu(self.gt),
                                  self.gt_obj, self.gt_obj_idx)
        gpu_res = self.get_result(cuda.to_gpu(self.input),
                                  cuda.to_gpu(self.gt),
                                  self.gt_obj, self.gt_obj_idx)
        gpu_res.to_cpu()
        numpy.testing.assert_almost_equal(cpu_res.data, gpu_res.data)

    def check_backward(self, x0_data, x1_data, y_grad):
        gradient_check.check_backward(
            functions.squared_error,
            (x0_data, x1_data), y_grad, eps=1e-2,
            **self.check_backward_options)


testing.run_module(__name__, __file__)
