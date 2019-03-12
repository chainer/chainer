import unittest

import numpy

import chainer
from chainer import configuration
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer.graph_optimizations.static_graph import static_graph
import chainer.links as L
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class StaticMLP(chainer.Chain):

    def __init__(self, in_size, n_out, W_dtype, x_dtype):
        super(StaticMLP, self).__init__()
        with self.init_scope():
            self.l1 = links.Linear(
                in_size, n_out,
                initialW=chainer.initializers.Normal(1, W_dtype),
                initial_bias=chainer.initializers.Normal(1, x_dtype))

    @static_graph(verbosity_level=2)
    def __call__(self, x):
        return F.relu(self.l1(x))


class DynamicMLP(chainer.Chain):

    def __init__(self, in_size, n_out, W_dtype, x_dtype):
        super(DynamicMLP, self).__init__()
        with self.init_scope():
            self.l1 = links.Linear(
                in_size, n_out,
                initialW=chainer.initializers.Normal(1, W_dtype),
                initial_bias=chainer.initializers.Normal(1, x_dtype))

    def __call__(self, x):
        return F.relu(self.l1(x))


class MLP(chainer.Chain):

    def __init__(self, in_size, n_out, W_dtype, x_dtype):
        super(MLP, self).__init__()
        with self.init_scope():
            initialW = chainer.initializers.Normal(1, W_dtype)
            initial_bias = chainer.initializers.Normal(1, x_dtype)
            self.l1 = links.Linear(in_size,
                                   n_out,
                                   initialW=initialW,
                                   initial_bias=initial_bias)
        self.mode = 'static'

    def __call__(self, x):
        if self.mode == 'static':
            return self.static_call(x)
        else:
            return self.dynamic_call(x)

    def dynamic_call(self, x):
        # Dynamic graph only.
        return F.relu(self.l1(x))

    @static_graph(verbosity_level=2)
    def static_call(self, x):
        # Static graph.
        return F.relu(self.l1(x))


@testing.parameterize(*testing.product({
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float32],
}))
class TestSimpleChain(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.in_units = 5
        self.out_units = 6
        x_size = (self.batch_size, self.in_units)
        self.x = numpy.random.uniform(size=x_size).astype(self.x_dtype)
        gy_size = (self.batch_size, self.out_units)
        self.gy = numpy.random.uniform(size=gy_size).astype(self.x_dtype)
        self.chain = MLP(self.in_units,
                         self.out_units,
                         self.W_dtype,
                         self.x_dtype)
        self.chain.l1.cleargrads()
        self.check_forward_options = {}
        self.check_backward_options = {'atol': 1e-2, 'rtol': 5e-2}
        self.dynamic_chain = DynamicMLP(self.in_units,
                                        self.out_units,
                                        self.W_dtype,
                                        self.x_dtype)
        self.static_chain = StaticMLP(self.in_units,
                                      self.out_units,
                                      self.W_dtype,
                                      self.x_dtype)

    def check_forward(self, x):
        y_dyn = self.chain.dynamic_call(x)
        with chainer.using_config('enable_backprop', False):
            y_static = self.chain.static_call(x)
            y_static = self.chain.static_call(x)
            y_static = self.chain.static_call(x)
        chainer.testing.assert_allclose(y_dyn.data, y_static.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    def test_forward_cpu2(self):
        y_dyn = self.chain.dynamic_call(self.x)
        x2 = 2*self.x
        # todo: add a new config so that we can still use 'train'
        with configuration.using_config('train', False):
            y_static1 = self.chain.static_call(x2)
            y_static1.grad = y_static1.data.copy()
            y_static1.backward()

            schedule_manager = self.chain.schedule_manager
            print('sched 1: ', schedule_manager)
            y_static = self.chain.static_call(self.x)
        chainer.testing.assert_allclose(y_dyn.data, y_static.data)

    @attr.gpu
    def test_forward_gpu(self):
        self.chain.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad, chain):
        gradient_check.check_backward(
            chain, x_data, y_grad, (chain.l1.W, chain.l1.b),
            dtype='f', **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        chain = self.static_chain
        with configuration.using_config('train', False):
            self.check_backward(self.x, self.gy, chain)


class MNISTStaticMLP(chainer.Chain):
    """This is the network from the MNIST example.

    Static version.
    """

    def __init__(self, n_units, n_out):
        super(MNISTStaticMLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    @static_graph(verbosity_level=2)
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class MNISTDynamicMLP(chainer.Chain):
    """This is the network from the MNIST example.

    Dynamic version.
    """

    def __init__(self, n_units, n_out):
        super(MNISTDynamicMLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


@testing.parameterize(*testing.product({
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
}))
class TestMultiLayerChain(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.in_units = 5
        self.out_units = 6
        self.hidden_units = 5
        x_size = (self.batch_size, self.in_units)
        self.x = numpy.random.uniform(size=x_size).astype(self.x_dtype)
        gy_size = (self.batch_size, self.out_units)
        self.gy = numpy.random.uniform(size=gy_size).astype(self.x_dtype)
        self.chain = MLP(self.in_units,
                         self.out_units,
                         self.W_dtype,
                         self.x_dtype)
        self.chain.l1.cleargrads()
        self.check_forward_options = {}
        self.check_backward_options = {'atol': 1e-2, 'rtol': 5e-2}
        self.dynamic_chain = MNISTDynamicMLP(self.hidden_units, self.out_units)
        self.static_chain = MNISTStaticMLP(self.hidden_units, self.out_units)

    def check_network_params_are_equal(self):
        static_W1_data = self.static_chain.l1.W.data
        dyn_W1_data = self.dynamic_chain.l1.W.data
        chainer.testing.assert_allclose(static_W1_data, dyn_W1_data)
        static_W2_data = self.static_chain.l2.W.data
        dyn_W2_data = self.dynamic_chain.l2.W.data
        chainer.testing.assert_allclose(static_W2_data, dyn_W2_data)
        static_W3_data = self.static_chain.l3.W.data
        dyn_W3_data = self.dynamic_chain.l3.W.data
        chainer.testing.assert_allclose(static_W3_data, dyn_W3_data)
        static_b1_data = self.static_chain.l1.b.data
        dyn_b1_data = self.dynamic_chain.l1.b.data
        chainer.testing.assert_allclose(static_b1_data, dyn_b1_data)
        static_b2_data = self.static_chain.l2.b.data
        dyn_b2_data = self.dynamic_chain.l2.b.data
        chainer.testing.assert_allclose(static_b2_data, dyn_b2_data)
        static_b3_data = self.static_chain.l3.b.data
        dyn_b3_data = self.dynamic_chain.l3.b.data
        chainer.testing.assert_allclose(static_b3_data, dyn_b3_data)

        static_W1_grad = self.static_chain.l1.W.grad
        dyn_W1_grad = self.dynamic_chain.l1.W.grad
        print('static_W1_grad: ', static_W1_grad)
        print('dyn_W1_grad: ', dyn_W1_grad)
        chainer.testing.assert_allclose(static_W1_grad, dyn_W1_grad)
        static_W2_grad = self.static_chain.l2.W.grad
        dyn_W2_grad = self.dynamic_chain.l2.W.grad
        chainer.testing.assert_allclose(static_W2_grad, dyn_W2_grad)
        static_W3_grad = self.static_chain.l3.W.grad
        dyn_W3_grad = self.dynamic_chain.l3.W.grad
        chainer.testing.assert_allclose(static_W3_grad, dyn_W3_grad)
        static_b1_grad = self.static_chain.l1.b.grad
        dyn_b1_grad = self.dynamic_chain.l1.b.grad
        chainer.testing.assert_allclose(static_b1_grad, dyn_b1_grad)
        static_b2_grad = self.static_chain.l2.b.grad
        dyn_b2_grad = self.dynamic_chain.l2.b.grad
        chainer.testing.assert_allclose(static_b2_grad, dyn_b2_grad)
        static_b3_grad = self.static_chain.l3.b.grad
        dyn_b3_grad = self.dynamic_chain.l3.b.grad
        chainer.testing.assert_allclose(static_b3_grad, dyn_b3_grad)

    def test_backward_custom_cpu(self):
        # Verify the both the Dynamic and Static networks produce the same
        # results on forward and backward passes.
        print('debug: Original input variable array: ', self.x)
        x_var_dyn = chainer.Variable(self.x)
        y_dyn = self.dynamic_chain(x_var_dyn)
        y_dyn.grad = self.gy
        y_dyn.backward()
        self.dynamic_chain.cleargrads()
        x_var_dyn.grad_var = None

        # Do forward and backward pass on the static chain and then
        # set its parameters to the same values as the dynamic chain.
        x_var_static = chainer.Variable(self.x.copy())
        y_static = self.static_chain(x_var_static)
        y_static.grad = self.gy
        y_static.backward()
        self.static_chain.cleargrads()
        x_var_static.grad_var = None
        self.static_chain.l1.W.data = self.dynamic_chain.l1.W.data.copy()
        self.static_chain.l1.b.data = self.dynamic_chain.l1.b.data.copy()
        self.static_chain.l2.W.data[...] = self.dynamic_chain.l2.W.data
        self.static_chain.l2.b.data[...] = self.dynamic_chain.l2.b.data
        self.static_chain.l3.W.data[...] = self.dynamic_chain.l3.W.data
        self.static_chain.l3.b.data[...] = self.dynamic_chain.l3.b.data

        # Do forward pass and verify that the outputs match the dynamic
        # chain.
        # Use a different input variable for this pass.
        x_size = (self.batch_size, self.in_units)
        new_x_data = numpy.random.uniform(size=x_size).astype(self.x_dtype)
        print('debug: 2nd iteration input variable array: ', new_x_data)
        x_var_dyn = chainer.Variable(new_x_data)
        x_var_static = chainer.Variable(new_x_data.copy())
        y_static = self.static_chain(x_var_static)
        assert y_static.data is not None
        y_dyn = self.dynamic_chain(x_var_dyn)
        assert y_dyn.data is not None
        chainer.testing.assert_allclose(y_dyn.data, y_static.data)

        # Use a different gy for the backward pass:
        y_size = (self.batch_size, self.out_units)
        new_y_data = numpy.random.uniform(size=y_size).astype(self.x_dtype)
        print('debug: 2nd iteration gy variable array: ', new_y_data)

        x_var_static.grad = None
        self.static_chain.cleargrads()

        y_static.grad = new_y_data
        y_static.backward()

        x_var_dyn.grad = None
        self.dynamic_chain.cleargrads()

        y_dyn.grad = new_y_data.copy()
        y_dyn.backward()
        assert x_var_dyn.grad is not None
        assert x_var_static.grad is not None
        chainer.testing.assert_allclose(x_var_dyn.grad, x_var_static.grad)

        self.check_network_params_are_equal()

        n_size = (self.batch_size, self.in_units)
        noise1 = 0.1*numpy.random.uniform(size=n_size).astype(self.x_dtype)
        x_pass1 = new_x_data + noise1

        # Modify l2.W's data:
        l2s = self.static_chain.l2.W.data.shape
        new_l2_W_data = 0.1*numpy.random.uniform(size=l2s).astype(self.x_dtype)
        self.static_chain.l2.W.data = new_l2_W_data
        self.dynamic_chain.l2.W.data = new_l2_W_data

        ns = (self.batch_size, self.out_units)
        new_y_data = numpy.random.uniform(size=ns).astype(self.x_dtype)

        x_var_static.data = x_pass1
        y_static = self.static_chain(x_var_static)
        assert y_static.data is not None
        y_static.grad = new_y_data
        self.static_chain.cleargrads()
        y_static.backward()

        x_var_dyn.data = x_pass1
        y_dyn = self.dynamic_chain(x_var_dyn)
        assert y_dyn.data is not None
        y_dyn.grad = new_y_data.copy()
        self.dynamic_chain.cleargrads()
        y_dyn.backward()
        chainer.testing.assert_allclose(y_dyn.data, y_static.data)

        self.check_network_params_are_equal()
        assert x_var_dyn.grad is not None
        assert x_var_static.grad is not None
        chainer.testing.assert_allclose(x_var_dyn.grad, x_var_static.grad)


testing.run_module(__name__, __file__)

if __name__ == '__main__':
    unittest.main()
