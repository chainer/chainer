import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import testing
from chainer import functions
from chainer.testing import attr

@testing.parameterize(*(testing.product({
    'delta_v': [0.5],
    'delta_d': [5],
    'alpha': [1],
    'beta': [1],
    'gamma': [0.001],
    'norm': [1],
    'result': [9709.743276179566]
}) + testing.product({
    'delta_v': [3],
    'delta_d': [10],
    'alpha': [0.1],
    'beta': [0.1],
    'gamma': [0.1],
    'max_n_clusters': [2],
    'norm': [2],
    'result': [2140420.229810252]
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
        chainer.config.type_check = False

    def get_result(self, x_data):
        return functions.discriminative_margin_based_clustering_loss(
            x_data, self.delta_v, self.delta_d,
            self.max_n_clusters, self.norm, self.alpha,
            self.beta, self.gamma)

    def check_forward_cpu(self, x_data, t_data):
        t = chainer.Variable(t_data)
        out = self.get_result(x_data)
        assert(out.data == t.data)

    def check_forward_gpu(self, x_data, t_data):
        t = chainer.Variable(t_data)
        out = self.get_result(x_data)
        out.to_cpu()
        t.to_cpu()

        assert (out.data == t.data)

    def test_forward_cpu(self):
        inputs = [cuda.to_cpu(self.input), cuda.to_cpu(self.gt),
                  self.gt_obj, self.gt_obj_idx]
        self.check_forward_cpu(inputs, self.y)

    @attr.gpu
    def test_forward_gpu(self):
        inputs = [cuda.to_gpu(self.input), cuda.to_gpu(self.gt),
                  self.gt_obj, self.gt_obj_idx]
        self.check_forward_gpu(inputs, self.y)

    @attr.gpu
    def test_forward_gpu_cpu(self):
        inputs_gpu = [cuda.to_gpu(self.input), cuda.to_gpu(self.gt),
                      self.gt_obj, self.gt_obj_idx]
        inputs_cpu = [cuda.to_cpu(self.input), cuda.to_cpu(self.gt),
                      self.gt_obj, self.gt_obj_idx]
        cpu_res = self.get_result(inputs_cpu)
        gpu_res = self.get_result(inputs_gpu)
        gpu_res.to_cpu()
        assert(cpu_res.data == gpu_res.data)


testing.run_module(__name__, __file__)
