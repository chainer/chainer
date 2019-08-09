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
    'result_l_dist': [185.15133466],
    'result_l_var': [64.81721673],
    'result_l_reg': [1.00000305]
}) + testing.product(({
    'delta_v': [3],
    'delta_d': [10],
    'alpha': [0.1],
    'beta': [0.1],
    'gamma': [0.1],
    'norm': [2],
    'result_l_dist': [0.59867187],
    'result_l_var': [27.54858666],
    'result_l_reg': [7.86060391]
})))
class TestDiscriminativeMarginBasedClusteringLoss(unittest.TestCase):

    def setUp(self):
        self.embedding_dims = 20
        self.batch = 5
        self.width = 128
        self.height = 128
        shape = (self.batch, self.embedding_dims,
                 self.height, self.width)

        input_arr = numpy.linspace(-100, 100,
                                   shape[1] * shape[2] * shape[3])
        self.input = input_arr.reshape((1, shape[1], shape[2], shape[3]))
        self.input = numpy.broadcast_to(self.input, shape)

        g_s = (self.batch, self.height, self.width)
        self.gt = numpy.zeros(g_s, dtype=numpy.int32)
        step_size = self.height // 10   # Create 10 different instances
        for b_idx in range(self.batch):
            for idx in range(10):
                self.gt[b_idx, (step_size * idx):(step_size * (idx + 1)), :] =\
                    b_idx * 10 + idx

        self.y = (numpy.asarray(self.result_l_dist),
                  numpy.asarray(self.result_l_var),
                  numpy.asarray(self.result_l_reg))

    def get_result(self, embeddings, labels):
        out = functions.discriminative_margin_based_clustering_loss(
            embeddings, labels, self.delta_v, self.delta_d,
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
