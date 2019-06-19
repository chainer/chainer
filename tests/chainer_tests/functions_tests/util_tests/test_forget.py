import functools
import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer import variable


@testing.parameterize(*testing.product({
    'out_len': [1, 2],
}))
class TestForget(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.y = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.gz0 = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.gz1 = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.ggx = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.ggy = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)

        self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
        self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-2}

    def check_forward(self, x_data, y_data):
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)
        if self.out_len == 1:
            z = functions.forget(lambda x, y: (x + y + x,), x, y)
            testing.assert_allclose(x_data + y_data + x_data, z.data)
        elif self.out_len == 2:
            z = functions.forget(lambda x, y: (x + y + x, x * y), x, y)
            testing.assert_allclose(x_data + y_data + x_data, z[0].data)
            testing.assert_allclose(x_data * y_data, z[1].data)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.y)

    def check_backward(self, x_data, y_data, *gz_data):
        def f(x, y):
            if self.out_len == 1:
                return functions.forget(lambda x, y: (x + y + x), x, y)
            elif self.out_len == 2:
                return functions.forget(lambda x, y: (x + y + x, x * y), x, y)

        gradient_check.check_backward(
            f, (x_data, y_data), gz_data, **self.check_backward_options)

    def test_backward_cpu(self):
        if self.out_len == 1:
            self.check_backward(self.x, self.y, self.gz0)
        elif self.out_len == 2:
            self.check_backward(self.x, self.y, self.gz0, self.gz1)

    @attr.gpu
    def test_backward_gpu(self):
        if self.out_len == 1:
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.y),
                                cuda.to_gpu(self.gz0))
        elif self.out_len == 2:
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.y),
                                cuda.to_gpu(self.gz0), cuda.to_gpu(self.gz1))


class TestForgetError(unittest.TestCase):

    def setUp(self):
        self.v = chainer.Variable(numpy.zeros(1))

    def test_not_callable(self):
        with self.assertRaises(TypeError):
            functions.forget(1)

    def test_invalid_type(self):
        with six.assertRaisesRegex(self, RuntimeError, 'int'):
            functions.forget(lambda: 1)

    def test_invalid_tuple_type_1st(self):
        with six.assertRaisesRegex(self, RuntimeError, '1st.*int'):
            functions.forget(lambda: (1,))

    def test_invalid_tuple_type_2nd(self):
        with six.assertRaisesRegex(self, RuntimeError, '2nd.*int'):
            functions.forget(lambda: (self.v, 1))

    def test_invalid_tuple_type_3rd(self):
        with six.assertRaisesRegex(self, RuntimeError, '3rd.*int'):
            functions.forget(lambda: (self.v, self.v, 1))

    def test_invalid_tuple_type_4th(self):
        with six.assertRaisesRegex(self, RuntimeError, '4th.*int'):
            functions.forget(lambda: (self.v,) * 3 + (1,))

    def test_invalid_tuple_type_11th(self):
        with six.assertRaisesRegex(self, RuntimeError, '11th.*int'):
            functions.forget(lambda: (self.v,) * 10 + (1,))

    def test_invalid_tuple_type_12th(self):
        with six.assertRaisesRegex(self, RuntimeError, '12th.*int'):
            functions.forget(lambda: (self.v,) * 11 + (1,))

    def test_invalid_tuple_type_13th(self):
        with six.assertRaisesRegex(self, RuntimeError, '13th.*int'):
            functions.forget(lambda: (self.v,) * 12 + (1,))


class TestForgetDoubleBackpropError(unittest.TestCase):

    def setUp(self):
        self.v = chainer.Variable(numpy.zeros(1))

    def test_invalid_double_backprop(self):
        with self.assertRaises(RuntimeError):
            x = functions.forget(lambda v: v, self.v)
            x.grad_var = variable.Variable(numpy.ones_like(x.data))
            x.backward(enable_double_backprop=True)


class TestForgetGrad(unittest.TestCase):

    def test_variable_grad(self):
        x = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        x = variable.Variable(x)
        w = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        w = variable.Variable(w)

        y = functions.forget(lambda a, b: a + b, x, w)
        y.grad_var = variable.Variable(numpy.ones_like(y.data))
        y.backward()

        assert isinstance(x.grad_var, variable.Variable)
        assert isinstance(w.grad_var, variable.Variable)

    def test_link_grad(self):

        class Model(chainer.link.Chain):
            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.link = links.Linear(None, 10)

            def forward(self, x):
                return functions.forget(self.link, x)

        model = Model()
        model.cleargrads()
        x = numpy.random.uniform(-1, 1, (64, 768)).astype(numpy.float32)
        x = variable.Variable(x, requires_grad=True)

        y = functions.sum(model(x))
        y.backward()
        assert isinstance(model.link.W.grad_var, variable.Variable)
        assert isinstance(model.link.b.grad_var, variable.Variable)


@testing.parameterize(*testing.product({
    'link_name': ['bn', 'brn', 'dbn'],
    'finetune': [False, True],
}))
class TestBatchNormalization(unittest.TestCase):

    def test_bn(self):

        class Model(chainer.link.Chain):
            def __init__(self, link_name, finetune, forget=False):
                super(Model, self).__init__()
                with self.init_scope():
                    if link_name == 'bn':
                        self.link = links.BatchNormalization(3)
                    elif link_name == 'brn':
                        # defaults rmax=1, dmax=0 are so trivial that BRN
                        # becomes BN
                        self.link = links.BatchRenormalization(
                            3, rmax=2.0, dmax=1.0)
                    elif link_name == 'dbn':
                        self.link = links.DecorrelatedBatchNormalization(
                            3, groups=3)
                self.forget = forget
                self.finetune = finetune

            def forward(self, x):
                if self.forget:
                    return functions.forget(
                        functools.partial(self.link, finetune=self.finetune),
                        x)
                else:
                    return self.link(x, finetune=self.finetune)

        x = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)

        model1 = Model(self.link_name, self.finetune, forget=False)
        model2 = Model(self.link_name, self.finetune, forget=True)

        # Update the models' internal statistics
        x1 = chainer.Variable(x)
        y = model1(x1)
        y.grad = gy
        y.backward()

        x2 = chainer.Variable(x)
        y = model2(x2)
        y.grad = gy
        y.backward()

        numpy.testing.assert_almost_equal(x1.grad, x2.grad)

        # Check if the outputs of the models are the same during test time
        with chainer.using_config('train', False):
            y1 = model1(x)
            y2 = model2(x)
            numpy.testing.assert_almost_equal(y1.data, y2.data)

        if self.finetune:
            assert model1.link.N == 1
            assert model2.link.N == 1


testing.run_module(__name__, __file__)
