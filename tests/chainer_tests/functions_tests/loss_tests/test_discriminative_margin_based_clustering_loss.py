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
    'result_l_dist': [6.0776708],
    'result_l_var': [64.0],
    'result_l_reg': [0.03419368]
}) + testing.product(({
    'delta_v': [3],
    'delta_d': [10],
    'alpha': [0.1],
    'beta': [0.1],
    'gamma': [0.1],
    'max_n_clusters': [2],
    'norm': [2],
    'result_l_dist': [0.0],
    'result_l_var': [26.56423595],
    'result_l_reg': [1.55665027]
})))
class TestDiscriminativeMarginBasedClusteringLoss(unittest.TestCase):

    def setUp(self):
        self.max_n_clusters = 5
        self.batch = 5
        self.width = 10
        self.height = 10
        shape = (self.batch, self.max_n_clusters,
                 self.width, self.height)

        input_arr = numpy.linspace(0, 100,
                                   shape[0] * shape[1] *
                                   shape[2] * shape[3])
        self.input = input_arr.reshape(shape)

        g_s = (self.batch, self.width, self.height)
        self.gt = numpy.linspace(0, 10,
                                 g_s[0] * g_s[1] * g_s[2]).astype(numpy.int32)
        self.gt = numpy.reshape(self.gt, g_s)
        self.y = (numpy.asarray(self.result_l_dist),
                  numpy.asarray(self.result_l_var),
                  numpy.asarray(self.result_l_reg))

    def get_result(self, embeddings, labels):
        out = functions.discriminative_margin_based_clustering_loss(
            embeddings, labels,
            self.delta_v, self.delta_d, self.max_n_clusters,
            self.norm, self.alpha, self.beta, self.gamma)
        return out

    def check_forward_cpu(self, embeddings, labels, t_data):
        t_dist, t_var, t_reg = \
            chainer.Variable(t_data[0]), \
            chainer.Variable(t_data[1]), \
            chainer.Variable(t_data[2])
        l_dist, l_var, l_reg = self.get_result(embeddings, labels)

        numpy.testing.assert_almost_equal(l_dist.data, t_dist.data)
        numpy.testing.assert_almost_equal(l_var.data, t_var.data)
        numpy.testing.assert_almost_equal(l_reg.data, t_reg.data)

    def check_forward_gpu(self, embeddings, labels, t_data):
        t_dist, t_var, t_reg = \
            chainer.Variable(t_data[0]), \
            chainer.Variable(t_data[1]), \
            chainer.Variable(t_data[2])
        l_dist, l_var, l_reg = self.get_result(embeddings, labels)
        l_dist.to_cpu()
        l_var.to_cpu()
        l_reg.to_cpu()
        t_dist.to_cpu()
        t_var.to_cpu()
        t_reg.to_cpu()
        numpy.testing.assert_almost_equal(l_dist.data, t_dist.data)
        numpy.testing.assert_almost_equal(l_var.data, t_var.data)
        numpy.testing.assert_almost_equal(l_reg.data, t_reg.data)

    def test_forward_cpu(self):
        self.check_forward_cpu(cuda.to_cpu(self.input), cuda.to_cpu(self.gt),
                               self.y)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward_gpu(cuda.to_gpu(self.input), cuda.to_gpu(self.gt),
                               self.y)

    @attr.gpu
    def test_forward_gpu_cpu(self):
        cpu_res = self.get_result(cuda.to_cpu(self.input),
                                  cuda.to_cpu(self.gt))
        gpu_res = self.get_result(cuda.to_gpu(self.input),
                                  cuda.to_gpu(self.gt))
        for idx in range(len(gpu_res)):
            gpu_res[idx].to_cpu()
            numpy.testing.assert_almost_equal(cpu_res[idx].data,
                                              gpu_res[idx].data)

    def check_backward(self, x0_data, x1_data, y_grad):
        gradient_check.check_backward(
            functions.squared_error,
            (x0_data, x1_data), y_grad, eps=1e-2,
            **self.check_backward_options)


testing.run_module(__name__, __file__)
