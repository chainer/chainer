import os
import tempfile
import unittest

import numpy

import chainer
from chainer import configuration
import chainer.functions as F
import chainer.links as L
from chainer.graph_optimizations.static_graph import static_graph
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer.serializers import npz
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


class StaticMLP(chainer.Chain):

    def __init__(self, in_size, n_out, W_dtype, x_dtype):
        super(StaticMLP, self).__init__()
        with self.init_scope():
            self.l1 = links.Linear(
                in_size, n_out,
                initialW=chainer.initializers.Normal(1, W_dtype),
                initial_bias=chainer.initializers.Normal(1, x_dtype))

    @static_graph(verbosity_level=0)
    def __call__(self, x):
        return self.l1(x)
        #return F.relu(self.l1(x))

class DynamicMLP(chainer.Chain):

    def __init__(self, in_size, n_out, W_dtype, x_dtype):
        super(DynamicMLP, self).__init__()
        with self.init_scope():
            self.l1 = links.Linear(
                in_size, n_out,
                initialW=chainer.initializers.Normal(1, W_dtype),
                initial_bias=chainer.initializers.Normal(1, x_dtype))

    def __call__(self, x):
        return self.l1(x)
        #return F.relu(self.l1(x))


class MLP(chainer.Chain):

    def __init__(self, in_size, n_out, W_dtype, x_dtype):
        super(MLP, self).__init__()
        with self.init_scope():
            #self.l1 = L.Linear(None, n_out)
            self.l1 = links.Linear(
            in_size, n_out,
            initialW=chainer.initializers.Normal(1, W_dtype),
            initial_bias=chainer.initializers.Normal(1, x_dtype))
        self.mode = 'static'
        #self.mode = 'dynamic'

    def __call__(self, x):
        if self.mode == 'static':
            return self.static_call(x)
        else:
            return self.dynamic_call(x)

    def dynamic_call(self, x):
        # Dynamic graph only.
        #return F.relu(self.l1(x))
        return self.l1(x)

    @static_graph(verbosity_level=0)
    def static_call(self, x):
        # Static graph.
        #return F.relu(self.l1(x))
        return self.l1(x)

@testing.parameterize(*testing.product({
    #'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
}))
class TestSimpleChain(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.in_units = 5
        self.out_units = 6
        self.x = numpy.random.uniform(size=(self.batch_size, self.in_units)).astype(self.x_dtype)
        self.gy = numpy.random.uniform(size=(self.batch_size, self.out_units)).astype(self.x_dtype)
        #print('x: ', self.x)
        self.chain = MLP(self.in_units, self.out_units, self.W_dtype, self.x_dtype)
        #W = self.chain.l1.W.data
        #b = self.chain.l1.b.data
        self.chain.l1.cleargrads()
        self.check_forward_options = {}
        self.check_backward_options = {'atol': 1e-2, 'rtol': 5e-2}
        self.dynamic_chain = DynamicMLP(self.in_units, self.out_units, self.W_dtype, self.x_dtype)
        self.static_chain = StaticMLP(self.in_units, self.out_units, self.W_dtype, self.x_dtype)

    def check_forward(self, x):
        y_dyn = self.chain.dynamic_call(x)
        with chainer.using_config('enable_backprop', False):
            y_static = self.chain.static_call(x)
            #schedule_manager = self.chain.schedule_manager
            #print("check_forward():schedule_manager: ", schedule_manager)
            y_static = self.chain.static_call(x)
            y_static = self.chain.static_call(x)
        #print('y_dyn \n', y_dyn)
        #print('y_static \n', y_static)
        chainer.testing.assert_allclose(y_dyn.data, y_static.data)
        #self.assertTrue(True)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    def test_forward_cpu2(self):
        y_dyn = self.chain.dynamic_call(self.x)
        x2 = 2*self.x
        with configuration.using_config('train', False):
        #with chainer.using_config('enable_backprop', True):
            y_static1 = self.chain.static_call(x2)
            y_static1.grad = y_static1.data.copy()
            y_static1.backward()

            schedule_manager = self.chain.schedule_manager
            print("sched 1: ", schedule_manager)
            #y_static = self.chain.static_call(x2)
            y_static = self.chain.static_call(self.x)
        # print('y_dyn \n', y_dyn)
        # print('y_static \n', y_static)
        chainer.testing.assert_allclose(y_dyn.data, y_static.data)

    @attr.gpu
    def tes_fixme_skipped_forward_gpu(self):
        self.chain.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad, chain):

        gradient_check.check_backward(
            chain, x_data, y_grad, (chain.l1.W, chain.l1.b),
            dtype='f', **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        # chain = self.chain
        #chain = self.dynamic_chain
        chain = self.static_chain

        #y = chain(self.x)
        #y.grad = self.gy
        #y.backward()
        #chain.cleargrads()

        #chainer.config.train = False
        with configuration.using_config('train', False):
            self.check_backward(self.x, self.gy, chain)


testing.run_module(__name__, __file__)

if __name__ == '__main__':
    unittest.main()